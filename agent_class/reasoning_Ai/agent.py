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
            "\n\nCRITICAL OUTPUT FORMATTING RULES:\n" + \
            "1. ABSOLUTELY NO markdown formatting: DO NOT use #, ##, ###, **, __, ~~ , -, *, +\n" + \
            "2. ABSOLUTELY NO special symbols that terminal cannot render: NO bullet points, NO arrows, NO brackets\n" + \
            "3. Write ONLY plain ASCII text and simple line breaks\n" + \
            "4. For lists, write as: Number one: ..., Number two: ..., etc\n" + \
            "5. For emphasis, use CAPS LOCK or write emphasis in parentheses (IMPORTANT: ...)\n" + \
            "6. Use only: periods, commas, colons, parentheses, numbers, and standard letters\n" + \
            "7. Every output MUST be safe to display in a terminal without any rendering issues\n" + \
            "\n\nRESPONSE FORMAT:\n" + \
            "Use this exact format for your thinking and actions:\n" + \
            "Thought: (explain your thinking here)\n" + \
            "Action: Search[query text here] OR Final Answer\n" + \
            "Keep this format consistent throughout."
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

            # 打印AI的思考和行動
            print(ai_output)

            # 將AI的輸出添加到消息歷史
            self.messages.append({"role": "assistant", "content": ai_output})

            # 解析 Action
            if "Action: Final Answer" in ai_output or "Action:Final Answer" in ai_output:
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
                print(f"\n[Search Query Detected: {search_query}]")

                # 執行搜索並等待結果
                print("[Executing search...]")
                search_result = tools.tavily_query(search_query, max_results=1)
                print("[Search completed]")

                # 構造 Observation 並讓AI處理
                observation_content = f"Observation: Here are the search results for '{search_query}':\n\n"
                observation_content += json.dumps(search_result,
                                                  ensure_ascii=False, indent=2)

                # 添加 Observation 到消息歷史，讓AI根據結果繼續思考
                self.messages.append(
                    {"role": "user", "content": observation_content})
                print(observation_content)
            else:
                # 沒有找到有效的 Action，可能是格式問題
                print("\n[Warning: Could not parse Action. Retrying...]")
                self.messages.append(
                    {"role": "user", "content": "Please provide your response in the correct format: Thought: ... then Action: Search[query] or Action: Final Answer"})

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
