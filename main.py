#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
Render (或任何要求"常驻进程监听 $PORT"的 PaaS)部署入口。

背景
----
scalp_engine.py 本身是一个"算完就退出"的 CLI 脚本:argparse 在启动
瞬间就要求 --symbol/--current/--target/--timeframe/--spread 这几个
必填参数,拿不到就 exit(2)。而 Render 的 Web Service 启动命令是不带
参数的 `python main.py`,期望进程常驻并在 $PORT 上接受 HTTP 请求——
两种进程模型不兼容,这就是之前部署日志里
"error: the following arguments are required" 后 "Exited with status 2"
的直接原因。

这个文件把 scalp_engine.py 的计算逻辑(run_engine / resolve_ticker)
包成一个最小的 Flask HTTP 服务,LLM 前端以后通过 HTTP 调用这里的
/analyze 接口,而不是 shell 出去跑 CLI 命令。

本地开发调试(CLI 模式)仍然直接用:
    python scalp_engine.py --symbol Silver --current 66 --target 66.5 \
        --timeframe 30 --spread 0.02 --mock

Render 部署时的 Start Command 改为:
    python main.py
或生产环境推荐用 gunicorn(见文末说明)。
"""

from __future__ import annotations

import os
import traceback

from flask import Flask, jsonify, request

from scalp_engine import run_engine, resolve_ticker, EngineResult

app = Flask(__name__)


# -------------------------------------------------------------- #
# 健康检查:Render 用它判断服务是否已就绪
# -------------------------------------------------------------- #
@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


@app.get("/")
def root():
    return jsonify({"service": "scalp_engine", "status": "ok", "endpoints": ["/analyze", "/healthz"]}), 200


# -------------------------------------------------------------- #
# 核心接口:LLM 前端通过这个接口拿计算结果
# -------------------------------------------------------------- #
@app.post("/analyze")
def analyze():
    """
    请求体 (JSON):
        {
          "symbol": "Silver",
          "current_price": 66.0,
          "target_price": 66.5,
          "timeframe_min": 30,
          "spread": 0.02,
          "interval": "1m",       // 可选,默认 "1m"
          "period": "1d",         // 可选,默认 "1d"
          "lookback_bars": 30,    // 可选,默认 30
          "mock": false           // 可选,默认 false;true 时用合成数据(联调/测试用,不接真实行情)
        }

    返回:
        200 -> run_engine() 计算出的标准 JSON(direction/hit_probability/...)
        400 -> 参数缺失或非法,body 里带 reason 说明具体缺了什么
        502 -> 计算过程本身出错(比如行情源取不到数据),body 里带 reason
    """
    body = request.get_json(silent=True) or {}

    # --- 必填参数校验(在这里就拦掉,而不是让 run_engine 内部抛出裸异常) ---
    required = ["symbol", "current_price", "target_price", "timeframe_min", "spread"]
    missing = [k for k in required if k not in body or body[k] is None]
    if missing:
        return (
            jsonify(
                {
                    "direction": "NO_TRADE",
                    "hit_probability": 0.0,
                    "tradeable": False,
                    "reason": f"缺少必填参数: {', '.join(missing)}",
                }
            ),
            400,
        )

    try:
        result: EngineResult = run_engine(
            symbol=str(body["symbol"]),
            current_price=float(body["current_price"]),
            target_price=float(body["target_price"]),
            timeframe_min=float(body["timeframe_min"]),
            spread=float(body["spread"]),
            interval=body.get("interval", "1m"),
            period=body.get("period", "1d"),
            lookback_bars=int(body.get("lookback_bars", 30)),
            mock=bool(body.get("mock", False)),
        )
        return app.response_class(result.to_json(), mimetype="application/json"), 200

    except ValueError as e:
        # 参数值本身非法(如 current_price<=0),400
        return (
            jsonify(
                {
                    "direction": "NO_TRADE",
                    "hit_probability": 0.0,
                    "tradeable": False,
                    "reason": f"参数非法: {str(e)}",
                }
            ),
            400,
        )
    except Exception as e:
        # 行情源异常/计算异常等,502,同时把 resolved_symbol 尽量带出来方便排查
        try:
            resolved = resolve_ticker(str(body.get("symbol", "")))
        except Exception:
            resolved = body.get("symbol")
        app.logger.error("‎analyze failed: %s\n%s", e, traceback.format_exc())
        return (
            jsonify(
                {
                    "direction": "NO_TRADE",
                    "hit_probability": 0.0,
                    "tradeable": False,
                    "reason": f"引擎异常: {str(e)}",
                    "resolved_symbol": resolved,
                }
            ),
            502,
        )


if __name__ == "__main__":
    # Render 通过环境变量 PORT 告诉容器应该监听哪个端口
    port = int(os.environ.get("PORT", 10000))
    # 生产环境建议换成 gunicorn(见 README / 文末说明),这里用 Flask 自带
    # 的开发服务器是为了让 `python main.py` 这条 Render 现有启动命令
    # 能直接工作,不用额外改 Start Command。
    app.run(host="0.0.0.0", port=port)
