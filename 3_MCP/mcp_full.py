from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Callable, List
import uuid
from datetime import datetime, timedelta
import requests
import json
from functools import wraps

# ==== 服务端实现 ====
app = FastAPI(
    title="模型上下文协议(MCP)服务",
    description="企业级LLM智能体交互协议服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class SessionCreateRequest(BaseModel):
    client_id: str = Field(..., description="客户端ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_timeout: int = Field(3600, description="会话超时时间(秒)")

class FunctionDefinition(BaseModel):
    name: str = Field(..., description="函数名称")
    description: str = Field(..., description="函数描述")
    parameters: List[Dict[str, Any]] = Field(..., description="参数列表")

class FunctionCallRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    function_name: str = Field(..., description="函数名称")
    parameters: Dict[str, Any] = Field(..., description="函数参数")

# 内存存储
sessions = {}
function_registry = {}

# 核心功能
def register_function(func: Callable, definition: FunctionDefinition):
    """注册函数到MCP服务"""
    function_registry[definition.name] = {
        "func": func,
        "definition": definition.dict()
    }

@app.post("/sessions", summary="创建新会话")
async def create_session(request: SessionCreateRequest):
    """创建新的MCP会话"""
    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(seconds=request.session_timeout)
    
    sessions[session_id] = {
        "session_id": session_id,
        "client_id": request.client_id,
        "user_id": request.user_id,
        "expires_at": expires_at,
        "timeout": request.session_timeout,
        "history": []
    }
    
    return {
        "session_id": session_id,
        "expires_at": expires_at,
        "status": "active"
    }

@app.post("/functions/call", summary="调用函数")
async def call_function(request: FunctionCallRequest):
    """调用注册的MCP函数"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
    
    if request.function_name not in function_registry:
        raise HTTPException(status_code=404, detail=f"Function {request.function_name} not registered")
    
    # 执行函数
    func_info = function_registry[request.function_name]
    try:
        result = func_info["func"](**request.parameters)
        
        # 记录调用历史
        sessions[request.session_id]["history"].append({
            "timestamp": datetime.utcnow(),
            "function": request.function_name,
            "parameters": request.parameters,
            "result": result
        })
        
        return {
            "session_id": request.session_id,
            "function_name": request.function_name,
            "result": result,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {
            "session_id": request.session_id,
            "function_name": request.function_name,
            "error": str(e),
            "timestamp": datetime.utcnow()
        }

# ==== 业务函数 ====
# 客户分群函数
def customer_segmentation(revenue: float, frequency: int, recency: int) -> str:
    """基于RFM模型的客户分群"""
    if recency < 30 and frequency > 12 and revenue > 10000:
        return "高价值客户"
    elif recency < 60 and frequency > 6 and revenue > 5000:
        return "增长型客户"
    elif recency > 180 and frequency < 3 and revenue < 1000:
        return "流失风险客户"
    else:
        return "一般价值客户"

# 销售预测函数
def sales_forecast(segment: str, budget: float) -> str:
    """基于客户分群和营销预算的销售预测"""
    base_growth = {
        "高价值客户": 0.15, "增长型客户": 0.25,
        "一般价值客户": 0.08, "流失风险客户": 0.30
    }
    growth_rate = base_growth.get(segment, 0.05)
    budget_impact = min(budget / 1000 * 0.01, 0.2)
    total_growth = growth_rate + budget_impact
    return f"{total_growth*100:.1f}%"

# 注册业务函数
register_function(
    func=customer_segmentation,
    definition=FunctionDefinition(
        name="customer_segmentation",
        description="基于RFM模型的客户分群函数",
        parameters=[
            {"name": "revenue", "type": "float", "description": "客户年度消费金额", "required": True},
            {"name": "frequency", "type": "int", "description": "客户年度购买频次", "required": True},
            {"name": "recency", "type": "int", "description": "最近购买天数", "required": True}
        ]
    )
)

register_function(
    func=sales_forecast,
    definition=FunctionDefinition(
        name="sales_forecast",
        description="基于客户分群和营销预算的销售预测函数",
        parameters=[
            {"name": "segment", "type": "string", "description": "客户分群结果", "required": True},
            {"name": "budget", "type": "float", "description": "营销预算金额", "required": True}
        ]
    )
)

# ==== 客户端实现 ====
class MCPClient:
    """MCP协议客户端"""
    
    def __init__(self, base_url: str, client_id: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.session_id = None
        self.headers = {"Content-Type": "application/json"}
    
    def create_session(self, user_id: Optional[str] = None, timeout: int = 3600) -> dict:
        """创建MCP会话"""
        response = requests.post(
            f"{self.base_url}/sessions",
            headers=self.headers,
            json={
                "client_id": self.client_id,
                "user_id": user_id,
                "session_timeout": timeout
            }
        )
        result = response.json()
        self.session_id = result.get("session_id")
        return result
    
    def call_function(self, function_name: str, parameters: Dict[str, Any]) -> dict:
        """调用MCP函数"""
        if not self.session_id:
            raise ValueError("请先创建会话")
            
        response = requests.post(
            f"{self.base_url}/functions/call",
            headers=self.headers,
            json={
                "session_id": self.session_id,
                "function_name": function_name,
                "parameters": parameters
            }
        )
        return response.json()

# ==== 启动服务 ====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
