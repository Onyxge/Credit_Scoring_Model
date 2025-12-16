from pydantic import BaseModel
from typing import Optional


class CreditRequest(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float

    Cat_airtime: Optional[int] = 0
    Cat_data_bundles: Optional[int] = 0
    Cat_financial_services: Optional[int] = 0
    Cat_movies: Optional[int] = 0
    Cat_other: Optional[int] = 0
    Cat_ticket: Optional[int] = 0
    Cat_transport: Optional[int] = 0
    Cat_tv: Optional[int] = 0
    Cat_utility_bill: Optional[int] = 0

    Ch_ChannelId_1: Optional[int] = 0
    Ch_ChannelId_2: Optional[int] = 0
    Ch_ChannelId_3: Optional[int] = 0
    Ch_ChannelId_5: Optional[int] = 0

    Avg_Tx_Hour: Optional[float] = 0
    Std_Tx_Hour: Optional[float] = 0
    Channel_Diversity: Optional[float] = 0
    Category_Diversity: Optional[float] = 0
    Engagement_Score: Optional[float] = 0


class CreditResponse(BaseModel):
    risk: int
    risk_probability: float
    label: str
    credit_score: int
