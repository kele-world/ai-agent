@echo off
chcp 65001 >nul
echo ========================================
echo   ESshop Agent - Docker 启动脚本
echo ========================================
echo.

where docker >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Docker
    echo 请先安装 Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo [错误] Docker 未运行，请先启动 Docker Desktop
    pause
    exit /b 1
)

if not exist .env (
    echo [提示] 未找到 .env，正在从模板创建...
    copy .env.example .env >nul
    echo.
    echo [重要] 请编辑 .env 填入 DEEPSEEK_API_KEY 后重新运行本脚本
    notepad .env
    pause
    exit /b 1
)

echo [1/2] 构建并启动容器...
docker compose up -d --build

echo.
echo [2/2] 等待服务就绪...
timeout /t 20 /nobreak >nul

echo.
echo ========================================
echo   服务状态
echo ========================================
docker compose ps

echo.
echo   访问地址: http://localhost:8100
echo   接口文档: http://localhost:8100/docs
echo   前端页面: 双击打开 index.html
echo.
echo   查看日志: docker compose logs -f agent
echo   停止服务: docker compose down
echo ========================================
pause
