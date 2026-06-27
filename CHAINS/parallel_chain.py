from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

myResume = """
Vishal Rana
8668343275 | job.vishalrana@gmail.com | Pune, India
LinkedIn | GitHub | Portfolio | Email
PERSONAL SUMMARY
Senior Android Developer with 8+ years of experience building scalable, high-performance mobile applications using Kotlin and Jetpack Compose. Strong expertise in clean architecture, banking, and enterprise applications.
TECHNICAL PROFICIENCY
 
•	Android Framework
•	Jetpack Compose
•	Kotlin, Java, C++
•	DI (Hilt, Koin)
•	Clean architecture, MVVM, MVI
•	Unit and UI test
•	User interface design
•	Code optimization
•	RoomDB and SQLite
•	HMI
•	Agile methodology
•	Jira
•	ChatGPT
•	Claude.AI, Github-Copilot
•	Tensorflow
•	Server Driven UI
•	Git, GitHub
•	Technical documentation
•	Team collaboration
•	Time management
•	Project research and management
•	Performance optimization 

EXPERIENCE
Advanced Semi Senior Engineer (Mobile Engineer), GLOBANT [Aug-2023 – Current]
Develop, test, and integrate a new application framework and recommend software improvements to ensure strong functionality and optimization.
Work with experienced team members to conduct root cause analysis of issues, review new and existing code, and/or perform unit testing.
Worked on LA Clippers + Intuit Dome App, Red Sea Global App, REI - Recreational Equipment, Inc. App 

Technology Analyst, INFOSYS LTD [Jan-2021 – Aug-2023]
Promoted to Tech analyst from junior developer position.
Make good technical decisions that provide solutions to business challenges. 
Develop, test, and integrate new application framework for Mobile banking and recommend software improvements to ensure strong functionality and optimization. 
Work with experienced team members to conduct root cause analysis of issues, review new and existing code and/or perform unit testing.
Worked on Kinetic Mobile [Banking App], HSBC Mobile Banking App


Software Developer, ALTIUS PVT LTD [Nov-2019 – Jan-2021]
Develop, test, and implement new software feature and deploy.
Test, maintain and recommend software improvements to ensure strong functionality and optimization. 
Worked on Lifecell International Projects, Babycord App 

Android Developer, APPSOFT INFOSYSTEM [Feb-2018 – Sep-2019]
Single handed manage the complete Android application Department and ongoing projects, with application hosting on play store.
Look out all the technical queries of the team and clients. Occupied with the development and support programme for different projects to modify the changes as per requirement of the client. 
Estimate level of effort, evaluate new options of similar technology, and offer suggestions to improve processes. 
Worked on iSync School ERP app, Staffing Connect job portal app

Web Handler, PHP WORKSHOP [Sep-2016 - Feb 2017]
Look out all the technical queries of the team and clients. Develop Various Website. 

EDUCATION
Bachelor of Engineering, IT, MIET Gondia Jan-2016
H.S.C DB Science College, Gondia Jan-2012

PERSONAL INFORMATION
Date of Birth: 27-Sep-1994 
Gender: Male 

LANGUAGES
Hindi, Marathi, English 

DISCLAIMER
I, Vishal Rana, hereby declare that the information contained herein is true and correct to the best of my knowledge and belief.

Date: 22-Jan-26

"""

parser = StrOutputParser()

skills_prompt = PromptTemplate.from_template(
    "Extract key skills from this resume:\n\n{resume}"
)

experience_prompt = PromptTemplate.from_template(
    "Extract work experience (role, company, years) from this resume:\n\n{resume}"
)

score_prompt = PromptTemplate.from_template(
    "Score this candidate out of 10 based on skills, experience and impact:\n\n{resume}"
)

summary_prompt = PromptTemplate.from_template(
    "Give a concise professional summary of this resume:\n\n{resume}"
)

# Chain 

chainSkill = skills_prompt | model | parser
chainExperience = experience_prompt | model | parser
chainScore = score_prompt | model | parser
chainSummary = summary_prompt | model | parser

parellelChain = RunnableParallel(
    skill = chainSkill,
    exp = chainExperience,
    score = chainScore,
    summary = chainSummary
)

result = parellelChain.invoke({'resume':myResume})

print(result)
print(parellelChain.get_graph().print_ascii())
