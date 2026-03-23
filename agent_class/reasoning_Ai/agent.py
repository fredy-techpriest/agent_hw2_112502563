import os
import re
import openai
import json
from dotenv import load_dotenv
import tools
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tool = [
    {
        "type": "function",
        "function": {
            "name": "tavily_query",
            "description": "search online for information related to the query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return."}
                },
                "required": ["query"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "provide the final answer to user's question",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string", "description": "The final answer to user's question."}
                },
                "required": ["answer"]
            }
        }
    }
]


class Agent:
    def __init__(self, system_prompt):
        # 增強系統提示，要求輸出純文本格式，不使用markdown
        # 同時定義 Thought/Action/Observation 格式
        enhanced_prompt = system_prompt + \
            "\n\nSTRICT OUTPUT CONTRACT:\n" + \
            "Return ONLY plain text and ONLY these lines:\n" + \
            "Line 1: Thought: <your reasoning>\n" + \
            "Line 2: Action: Search[<single search query>] OR Action: Final Answer\n" + \
            "Optional line 3 (only when line 2 is Action: Final Answer): Answer: <final answer>\n" + \
            "Do not output any other prefixes such as Final Answer:, Observation:, Plan:, or Note:.\n" + \
            "Do not use markdown or bullet points in your output.\n" + \
            "Brackets [] are REQUIRED only for Search action.\n" + \
            "If your previous output had invalid format, immediately retry with the exact contract."
        self.system = enhanced_prompt
        self.messages = []
        self.messages.append({"role": "system", "content": self.system})

    def construct_prompt(self, query):
        self.messages.append({"role": "system", "content": self.system})

    def execute(self, query):
        # The ReAct Loop
        self.messages.append({"role": "user", "content": query})
        iteration = 0

        while iteration < 5:
            print("\n" + "="*80)
            print(f"ITERATION {iteration}")
            print("="*80)

            # 讓AI生成 Thought 和 Action
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages
            )
            response_msg = response.choices[0].message
            ai_output = response_msg.content

            is_final_action = "Action: Final Answer" in ai_output or "Action:Final Answer" in ai_output

            # 最終答案回合不先打印，避免與外層 print(agent.execute(...)) 重複
            if not is_final_action:
                print(ai_output)

            # 將AI的輸出添加到消息歷史
            self.messages.append({"role": "assistant", "content": ai_output})

            # 解析 Action
            if is_final_action:
                # 最終回合仍顯示 Thought，避免使用者看不到推理狀態
                for line in ai_output.split("\n"):
                    if line.strip().startswith("Thought:"):
                        print(line)
                        break

                # AI認為有足夠信息提供最終答案
                print("\n[FINAL ANSWER]")
                print("="*80)
                # 提取最終答案（在 Thought/Action 之後的內容）
                lines = ai_output.split("\n")
                answer_started = False
                answer_lines = []
                for line in lines:
                    if "Action:" in line:
                        answer_started = True
                        continue
                    if answer_started and line.strip():
                        answer_lines.append(line)

                if answer_lines:
                    return "\n".join(answer_lines)
                else:
                    return ai_output

            # 解析搜索查詢
            import re
            search_match = re.search(r'Action:\s*Search\[(.*?)\]', ai_output)
            if search_match:
                search_query = search_match.group(1)

                # 執行搜索並等待結果
                search_result = tools.tavily_query(search_query, max_results=1)

                # 構造 observation：只保留 content
                contents = [item.get("content", "").strip()
                            for item in search_result.get("results", [])
                            if item.get("content", "").strip()]
                if not contents and search_result.get("answer", "").strip():
                    contents = [search_result.get("answer", "").strip()]

                merged_content = "\n\n".join(
                    contents) if contents else "No relevant content found."
                observation_content = f"[observation]\n{merged_content}"

                # 添加 Observation 到消息歷史，讓AI根據結果繼續思考
                self.messages.append(
                    {"role": "user", "content": observation_content})
                print(observation_content)
            else:
                # 沒有找到有效的 Action，可能是格式問題
                print("\n[Warning: Could not parse Action. Retrying...]")
                self.messages.append(
                    {"role": "user", "content": "Invalid format. Retry exactly with: Line1 Thought: ... Line2 Action: Search[query] or Action: Final Answer. If Final Answer, add Line3 Answer: ..."})

            iteration += 1

        # 超過5次循環
        print("\n[Maximum iterations reached]")
        print("\n[FINAL ANSWER]")
        print("="*80)
        self.messages.append(
            {"role": "user", "content": "You have reached the maximum number of iterations. Please provide your final answer now."})

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        final_msg = final_response.choices[0].message
        return final_msg.content
