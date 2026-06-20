from langchain_openai import ChatOpenAI
from typing import Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Schema
class UserInfo(BaseModel):
    key_themes: list[str] = Field(..., description="List down all Key themes user added in there review")
    summary: str = Field(..., description="A brief summary of the user's information")
    age: Optional[int] = Field(None, description="The user's age")
    email: Optional[str] = Field(None, description="The user's email address")
    sentiment: Literal["positive", "negative"] = Field(..., description="Indicates the user's sentiment")
    pros: list[str] = Field(..., description="A list of positive attributes")
    cons: list[str] = Field(..., description="A list of the user's negative attributes")
    name: Optional[str] = Field(None, description="The reviewer's name")

structuralModel = ChatOpenAI().with_structured_output(UserInfo) # or ChatOpenAI(model="gpt-3.5-turbo", response_format=UserInfo)

response = structuralModel.invoke(
    """
Moto G67 Review

By Vishal Rana

The Motorola Moto G67 Smartphone is a budget-friendly smartphone that focuses on delivering a large display, strong battery life, and clean Android experience. It comes with a 6.78-inch 120Hz display, MediaTek Dimensity 6300 processor, and a 5200mAh battery with 30W fast charging, making it suitable for everyday users.

In real-world usage, the Moto G67 performs well for daily tasks like social media, calling, browsing, and light gaming, but it is not designed for heavy performance or high-end gaming. The camera setup (50MP + 8MP) delivers decent daylight shots, while low-light performance is average.

Overall, the Moto G67 is a balanced budget phone for users who want reliability and battery life over raw power.

Pros
Excellent Battery Life
5200mAh battery easily lasts a full day or more with normal usage.
Big & Smooth Display
6.78-inch display with 120Hz refresh rate gives smooth scrolling and good viewing experience.
Good Daylight Camera
50MP main camera captures sharp and clear photos in good lighting conditions.
Clean Android Experience
Runs Android 16 with minimal bloatware (Motorola’s near-stock UI).
Durable Design
IP64 rating provides protection against dust and splashes.

Cons
Average Performance
Dimensity 6300 is fine for daily use but struggles with heavy gaming or multitasking.
Low RAM (4GB)
Limited RAM may cause lag with multiple apps.
Weak Low-Light Camera
Night photography is not very impressive compared to competitors
Average Speakers
Sound quality is decent but lacks bass and richness (as per user feedback)
Limited Software Updates
Motorola usually offers fewer updates compared to brands like Samsung or Xiaomi
"""
)


print(response.sentiment)  # Accessing the sentiment field from the response
