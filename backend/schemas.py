
from pydantic import BaseModel, Field


class DemandPredictionRequest(BaseModel):
    Category: str
    Region: str

    inventory_level: float = Field(alias="Inventory Level")
    Price: float
    Discount: float
    weather_condition: str = Field(alias="Weather Condition")
    Promotion: str
    competitor_pricing: float = Field(alias="Competitor Pricing")
    Seasonality: str
    Epidemic: str

    Year: int
    Month: int
    Day: int
    DayOfWeek: int

    Price_Difference: float
    Price_Difference_Percent: float
    Stock_Status: str
    Inventory_to_Order_Ratio: float

    model_config = {"populate_by_name": True}


class DemandPredictionResponse(BaseModel):
    predicted_demand: float