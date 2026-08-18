from pydantic import BaseModel, Field


class ApiResponse[DataT](BaseModel):
    """所有成功接口使用的统一响应结构。"""

    code: int = Field(default=0, description="业务状态码，0 表示成功")
    message: str = Field(default="success", description="响应说明")
    data: DataT


class HealthData(BaseModel):
    """健康检查数据。"""

    status: str
    service: str
