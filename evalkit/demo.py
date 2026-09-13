"""演示被测应用（Phase 0）：两个规则实现的迷你 AI 应用，用于验证通用引擎。

它们不是玩具摆设，而是引擎的"对照实验组"：
- DemoChatAdapter：普通聊天应用（关键词 FAQ + 边界拒答）→ 验证 chat_profile；
- DemoJsonAdapter：结构化抽取应用（正则抽字段）→ 验证 json_profile / JSON Schema 评分。

全部确定性、零 LLM 成本、离线可跑——CI 和冒烟都依赖它们。
真实 LLM 应用的接入方式完全相同：实现 TargetAdapter.invoke 即可。
"""
from __future__ import annotations

import json
import re

from evalkit.contracts import InvocationRequest, TargetObservation, TargetAdapter


class DemoChatAdapter(TargetAdapter):
    """迷你售后客服：命中关键词答 FAQ，越界问题礼貌拒答。"""

    target_id = "demo-chat"

    KB = [
        ("退款", "我们支持 7 天无理由退款，请在订单页点击「申请退款」，1-3 个工作日原路退回。"),
        ("发货", "现货商品下单后 48 小时内发货，偏远地区略有顺延。"),
        ("发票", "支持开具电子发票，下单时在备注填写抬头即可。"),
        ("换货", "收货 15 天内可申请同型号换货，来回运费由商家承担。"),
    ]
    REFUSAL = "抱歉，我只能回答售后相关的问题。"

    def invoke(self, request: InvocationRequest) -> TargetObservation:
        text = request.text()
        for keyword, answer in self.KB:
            if keyword in text:
                return TargetObservation(status="success", output=answer)
        if any(word in text for word in ("骂", "打你", "攻击")):
            return TargetObservation(status="rejected", output="我不能回应这类内容。")
        return TargetObservation(status="success", output=self.REFUSAL)


class DemoJsonAdapter(TargetAdapter):
    """迷你信息抽取：从文本抽 {name, age}，字段缺失时就地缺省（供 Schema 评分抓出）。"""

    target_id = "demo-json"

    def invoke(self, request: InvocationRequest) -> TargetObservation:
        text = request.text()
        out: dict = {}
        name = re.search(r"姓名[:：]?\s*([\u4e00-\u9fa5]{2,4})", text)
        if name:
            out["name"] = name.group(1)
        age = re.search(r"年龄[:：]?\s*(\d{1,3})", text) or re.search(r"(\d{1,3})\s*岁", text)
        if age:
            out["age"] = int(age.group(1))
        return TargetObservation(
            status="success",
            output=json.dumps(out, ensure_ascii=False),
            structured_output=out,
        )
