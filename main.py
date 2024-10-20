# API Imports
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel

# LLM imports
from langchain.chains.flare.prompts import PROMPT_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from rag import getDB, generate_data_store
from consts import CHROMA_PATH
import asyncio


class state:
    doctor_id = 63902541323
    transcript_text = ""
    current_appointment = ""
    dirty = False
    kill = False
    task = None

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
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
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
    file_path = os.path.join(
        "./reports/", DeleteReportRequest.appointment_id+".json")

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
    file_path = os.path.join(
        "./reports/", UpdateReportRequest.appointment_id+".json")

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
        "doctor_id": UpdateReportRequest.doctor_id,
        "appointment_id": UpdateReportRequest.appointment_id,
        "date": UpdateReportRequest.date,
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

    # Run Agent Handler in new Thread
    asyncio.create_task(agent_handler())
    return {"appointment_id": appointment_id}


class AppointmentEndRequest(BaseModel):
    doctor_id: int  # Required field
    appointment_id: str


@app.post("/endAppointment")
def endAppointment(AppointmentEndRequest: AppointmentEndRequest):
    if state.doctor_id != AppointmentEndRequest.doctor_id:
        raise HTTPException(
            status_code=400, detail="Failure: Doctor ID mismatch")

    if state.current_appointment != AppointmentEndRequest.appointment_id:
        raise HTTPException(
            status_code=400, detail="Failure: Appointment ID mismatch")

    # TODO: Generate data based on current state
    data = {
        "doctor_id": state.doctor_id,
        "appointment_id": state.current_appointment,
        "date": datetime.now().strftime("%B %d, %Y"),
        "data": {
            "name": "Alan Wang",
            "age": "23",
            "chief_complaint": "Itchy",
            "history_of_present_illness": "COVID",
            "family_history": "Byron Wang",
            "social_history": "Simba Hu",
            "review_of_symptoms": "Lol"
        }
    }

    file_name = "./reports/"+data["appointment_id"]+".json"

    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except:
        print(f"{file_name}: Error Writing JSON")

    state.kill = True
    return {"message": "Appointment Ended"}


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
    return {"data": state.windows}


# RAG pipeline
@app.post("/generateDatastore")
def generateDatastore():
    try:
        generate_data_store()
    except Exception as e:
        print(e)

    return {"message": "Database Generated Successfully"}


# Async Agent Calls
async def create_summary():
    # bullet points
    return False


async def create_detailed_summary():
    # use perplexity
    # ideally rag step here
    # mix of diff things to look out for + diagnosis + analyze entire convo
    return False


async def conduct_background_research():
    # gpt call to do more research / grok
    return False


async def create_dynamic_ui():
    # anthropic llm / gpt4 (give it all info from 3 calls)
    # smartly split it into 9 windows in specific format
    state.windows = [{"id": 1, "data": "Bruh"}, {
        "id": 2, "data": "Bruh"}, {"id": 3, "data": "Bruh"}]
    return False


async def agent_handler():
    while True:
        if (not state.dirty):
            continue
        if (state.kill):
            print(f"Async Thread Killed")
            return

        # Create Simple Summary
        await create_summary()

        # Create More Complex Summaries with RAG and Perplexity Search
        await create_detailed_summary()

        # Do extra Background research
        await conduct_background_research()

        # Use LLMs to generate UI
        await create_dynamic_ui()

        print(f"Updated Summaries")
        await asyncio.sleep(5)  # Wait for 5 seconds before checking again
