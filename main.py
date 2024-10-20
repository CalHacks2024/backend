# API Imports
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel
from groq import Groq

# LLM imports
from langchain.chains.flare.prompts import PROMPT_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate
from openai import Completion
from rag import getDB, generate_data_store
from consts import CHROMA_PATH
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
# from langchain.schema.runnable import RunnableSequence
from langchain_core.runnables import RunnableSequence, RunnableMap
import asyncio

class state:
    doctor_id = 63902541323
    transcript_text = ""
    current_appointment = ""
    dirty = False
    kill = False
    task = None
    groqSimple = ""
    groqDetailed = ""
    questions = ""
    diagnosis = ""
    moreResearchRequired = ""

    windows = {}


# Init API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (including custom headers)
)

state = state()
# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )
groqClient = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

getDB()



# Health Check
@app.get("/healthCheck")
def healthCheck():
    return {"message": "Server Alive"}


# Report Routes
@app.get("/reports")
def reports(doctor_id: int):
    directory = './reports'

    res = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):  # Check for JSON files
            file_path = os.path.join(directory, filename)
            
            try:
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    # Check if 'doctor_id' exists in the JSON data
                    if 'doctor_id' in data and data['doctor_id'] == doctor_id:
                        res.append(data)
      
            except json.JSONDecodeError:
                print(f"{filename}: Error decoding JSON")
            except Exception as e:
                print(f"{filename}: Error reading file - {e}")

    return {"reports": res}

class DeleteReportRequest(BaseModel):
    appointment_id: str

@app.delete("/deleteAppointment")
def deleteAppointment(DeleteReportRequest: DeleteReportRequest):
    # lmao super unsafe but this is an mvp so ¯\_(ツ)_/¯
    file_path = os.path.join("./reports/", DeleteReportRequest.appointment_id+".json")
    
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Delete the file
        os.remove(file_path)
        return {"detail": "File deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class Report(BaseModel):
    name: str
    age: str
    chief_complaint: str
    history_of_present_illness: str
    family_history: str
    social_history: str
    review_of_symptoms: str

    
class UpdateReportRequest(BaseModel):
    doctor_id: int
    appointment_id: str
    date: str
    data: Report

@app.post("/updateAppointment")
def updateAppointment(UpdateReportRequest: UpdateReportRequest):
    # lmao super unsafe but this is an mvp so ¯\_(ツ)_/¯
    file_path = os.path.join("./reports/", UpdateReportRequest.appointment_id+".json")
    
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Delete the file
        os.remove(file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    file_name = "./reports/" + UpdateReportRequest.appointment_id + ".json"
    data = {
        "doctor_id":UpdateReportRequest.doctor_id,
        "appointment_id":UpdateReportRequest.appointment_id,
        "date":UpdateReportRequest.date,
        "data": {
            "name": UpdateReportRequest.data.name,
            "age": UpdateReportRequest.data.age,
            "chief_complaint": UpdateReportRequest.data.chief_complaint,
            "history_of_present_illness": UpdateReportRequest.data.history_of_present_illness,
            "family_history": UpdateReportRequest.data.family_history,
            "social_history": UpdateReportRequest.data.social_history,
            "review_of_symptoms": UpdateReportRequest.data.review_of_symptoms
        }
    }
    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        print(e)

    return {"detail": "File updated successfully"}


# Spectacles Appointments 
class DoctorRequest(BaseModel):
    doctor_id: int  # Required field

@app.post("/startAppointment")
async def startAppointment(DoctorRequest: DoctorRequest):
    state.doctor_id = int(DoctorRequest.doctor_id)

    # Generate a UUID1
    appointment_id = uuid.uuid1()
    state.current_appointment = str(appointment_id)
    state.kill = False
    
    ## Run Agent Handlers in new Thread
    asyncio.create_task(agent_handler())
    return {"appointment_id":appointment_id}


class AppointmentEndRequest(BaseModel):
    doctor_id: int  # Required field
    appointment_id: str


def generate_report(information):
    prompt = f"""
    Convert the following information about a patient into an obejct with the fields:
    chief_complaint, history_of_present_illness, family_history, social_history and review_of_symptoms.
    Avoid complex medical terminology and be concise.

    Information: {information}

    Output the structured result in a JSON format with the specified fields. Below is a description of each field:
    chief_complaint: the patient's main issues in their own words.
    history_of_present_illness: any relevant medical history related to the issue.
    family_history: any relevant family information related to the issue.
    social_history: any relevant information about exposure to others, travel or infectious diseases.
    review_of_symptoms: a general summary of associated signs and symptoms.
    """

    response = Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=300,
        temperature=0,
        stop=["Output"]
    )

    structured_output = response['choices'][0]['text'].strip()
    return structured_output

@app.post("/endAppointment")
def endAppointment(AppointmentEndRequest: AppointmentEndRequest):
    if state.doctor_id != AppointmentEndRequest.doctor_id:
        raise HTTPException(status_code=400, detail="Failure: Doctor ID mismatch") 
    
    if state.current_appointment != AppointmentEndRequest.appointment_id:
        raise HTTPException(
            status_code=400, detail="Failure: Appointment ID mismatch")

    report = generate_report(state.groqDetailed)
    # TODO: Generate data based on current state

    data = {
        "doctor_id":state.doctor_id,
        "appointment_id":state.current_appointment,
        "date":datetime.now().strftime("%B %d, %Y"),
        "data": {
            "name": "Alan Wang",
            "age": "23",
            "chief_complaint": report.chief_complaint,
            "history_of_present_illness": report.history_of_present_illness,
            "family_history": report.family_history,
            "social_history": report.social_history,
            "review_of_symptoms": report.review_of_symptoms
        }
    }

    file_name = "./reports/"+data["appointment_id"]+".json"

    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except:
        print(f"{file_name}: Error Writing JSON")
    
    state.kill = True
    return {"message":"Appointment Ended"}


# Spectacles Transcript
class TranscriptRequest(BaseModel):
    appointment_id: str
    transcript: str

@app.post("/sendTranscript")
def sendTranscript(TranscriptRequest: TranscriptRequest):
    # if state.current_appointment != TranscriptRequest.appointment_id:
    #     raise HTTPException(status_code=400, detail="Failure: Appointment ID mismatch") 
    
    state.transcript_text = TranscriptRequest.transcript
    state.dirty = True
    print(state.transcript_text)
    return {"data":state.windows}


# RAG pipeline
@app.post("/generateDatastore")
def generateDatastore():
    try:
        generate_data_store()   
    except Exception as e:
        print(e)

    return {"message":"Database Generated Successfully"}


# Async Agent Calls
async def create_summary():
    # grok or anthropic (claude)
    chat_completion = groqClient.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a quick bullet point summary of the contents: " + state.transcript_text
            }
        ],
        model="llama3-8b-8192"
    )
    print("quick summary is the following: ")
    print(chat_completion.choices[0].message.content)
    state.groqSimple = chat_completion.choices[0].message.content
    return False

async def create_detailed_summary():
    # perplexity
    chat_completion = groqClient.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "You are a medical professional responsible for evaluating patients and providing appropriate care recommendations. You are tasked with assessing patient information to determine the need for further investigation or treatment. You have access to the patient's medical history, which contains detailed information about their symptoms, previous diagnoses, and treatment responses. You also have access to clinical guidelines and a medical database with evidence-based articles. Additionally, you can refer to a network of healthcare professionals for collaborative insights. Unless you are confident in your diagnosis or treatment plan, refer the patient to a specialist or another healthcare provider for further evaluation using the escalateToHuman function. Do not provide uncertain or unverified medical advice. The patient's information is as follows. Your goal is to assist the patient in achieving their health objectives to the best of your ability. Keep it concise, do not be too wordy. Give your result in 3rd person + keep it in CONCISE POINTS. : " + state.transcript_text
            }
        ],
        model="llama3-8b-8192"
    )
    print("quick summary is the following: ")
    print(chat_completion.choices[0].message.content)
    state.groqDetailed = chat_completion.choices[0].message.content
    return False

def conduct_background_research():
    print("starting the conducting")
    
    llm = ChatOpenAI(model="gpt-4o")

    transcript_text = state.groqDetailed

    prompt = PromptTemplate(
        input_variables=["transcript_text"],
        template="Analyze these contents: {transcript_text}, tell me either True or False if this is detailed enough or if I should do more research, you can be more lenient. Again, ONLY tell me 'True' or 'False'"
    )

    # Use RunnableSequence to chain the prompt and LLM
    chain = RunnableSequence(prompt | llm)

    # Conduct research
    detailed_summary_check = chain.invoke({"transcript_text": transcript_text})

    # Check if more research is required
    print(detailed_summary_check.content, ' is the thing byron')
    state.moreResearchRequired = detailed_summary_check.content == "True"
    # state.questions = detailed_summary_check.
    print(f"{state.moreResearchRequired} is the CHECK")

    return state.moreResearchRequired 

async def create_dynamic_ui():
    # 1 - summary point form
    # 2 - general facts from patient that will support diagnosis
    # 3 - questions that can be asked
    # 4 - summary of current potential diagnosis
    # type res = {
        # id: string;
        # title: string;
        # body: string;
    #}
    windowContent = [
        { "id": 1, "title": "Summary point form", "body": state.groqSimple },
        { "id": 2, "title": "General facts from the patient", "body": state.groqDetailed },
        { "id": 3, "title": "Questions you can ask the patient", "body": state.moreResearchRequired.questions },
        { "id": 4, "title": "Potential diagnosis", "body": state.diagnosis },
    ]

    # state.windows = [{"id":1,"data":"Bruh"},{"id":2,"data":"Bruh"},{"id":3,"data":"Bruh"}]
    state.windows = windowContent
    return False


async def agent_handler():
    while True:
        if (not state.dirty):
            await asyncio.sleep(10)  # Wait for 5 seconds before checking agai
            continue

        if(state.kill):
            print(f"Async Thread Killed")
            return
        
        # Create Simple Summary
        await create_summary()

        # Create More Complex Summaries with RAG and Perplexity Search
        await create_detailed_summary()

        # Do extra Background research
        print("beforehand byron")
        conduct_background_research()

        # Use LLMs to generate UI
        await create_dynamic_ui()

        print(f"Updated Summaries")
        state.dirty = False
        await asyncio.sleep(5)  # Wait for 5 seconds before checking agai
    