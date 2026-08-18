from fastapi import APIRouter

from app.schemas.response import ApiResponse, HealthData

router = APIRouter()


@router.get(
    "/health",
    response_model=ApiResponse[HealthData],
    summary="检查服务运行状态",
)
async def health_check() -> ApiResponse[HealthData]:
    """返回当前服务的基本运行状态。"""
    return ApiResponse(
        data=HealthData(
            status="healthy",
            service="智能故障诊断助手",
        )
    )
