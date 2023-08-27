import openai, os, time, sys

if len(sys.argv) < 2:
    print("usage: python {} num".format(sys.argv[0]))
    sys.exit(1)

openai.api_key = os.getenv('OPENAI_API_KEY')
i = 0
backoff = 1
while i < int(sys.argv[1]): 
    print(">>>>>>>>>>>>>>>>>>>>>>>> {}".format(i+1))
    try:
        o = openai.ChatCompletion.create(
            model = "gpt-4",
    messages = [
                {"role": "system", "content":"너는 경제신문 기자야."},
                {"role": "user", "content":"""아래 조건들을 만족시키면서 인플레이션 또는 물가와 상관이 없는 기사 문장 5개를 생성해줘. 
조건 1. 기사 문장은 길 수도 있고 짧을 수도 있어. 
조건 2. 기사 문장에는 수치값을 포함할 수 있고, 안 해도 돼"""
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
