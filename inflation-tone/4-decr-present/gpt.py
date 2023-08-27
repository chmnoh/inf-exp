import openai, os, time, sys

if len(sys.argv) < 2:
    print("usage: python {} num".format(sys.argv[0]))
    sys.exit(1)

if not os.path.exists('data'):
    os.mkdir('data')

os.makedirs("data", exist_ok=True)
openai.api_key = os.getenv('OPENAI_API_KEY')
i = 0
backoff = 1
while i < int(sys.argv[1]): 
    print(">>>>>>>>>>>>>>>>>>>>>>>> {}".format(i+1))
    try:
        o = openai.ChatCompletion.create(
            model = "gpt-4",
            # model = "gpt-3.5-turbo-16k",
            messages = [
                {"role": "system", "content":"너는 경제신문 기자야."},
                {"role": "user", "content":"""아래 조건들을 만족시키면서 인플레이션이 완화됐다는 기사 문장 5개를 생성해줘. 
조건 1. 기사 문장은 길 수도 있고 짧을 수도 있어. 
조건 2. 기사 문장이 금리나 환율 등 물가에 간접적으로 영향을 주는 요소들도 언급할 수 있어. 
조건 3. 기사 문장에서 "인플레이션", "물가", "가격" 같은 단어는 제외해줘. 
조건 4. 기사 문장에는 수치값도 포함해야 해."""
                }
            ],
            temperature = 1.0
        )
        i += 1
        print(o.choices[0].message.content)
        with open("data/{}.txt".format(time.strftime("%Y-%m-%d-%H%M%S")), "a+") as f:
            f.write(o.choices[0].message.content)
    except openai.error.APIError as e:
        #Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        #Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.error.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        time.sleep(backoff)
        backoff *= 2
        pass
    except:
        time.sleep(5.0)
        pass

#
# vim:sw=4:ts=4:expandtab:
# 
