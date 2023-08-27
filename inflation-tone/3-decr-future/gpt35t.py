import openai, os, time, sys

if len(sys.argv) < 2:
    print("usage: python {} num".format(sys.argv[0]))
    sys.exit(1)

if not os.path.exists('data'):
    os.mkdir('data')

openai.api_key = os.getenv('OPENAI_API_KEY')
i = 0
backoff = 1
while i < int(sys.argv[1]): 
    print(">>>>>>>>>>>>>>>>>>>>>>>> {}".format(i+1))
    try:
        o = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content":"너는 유능하면서 경제 전문지식이 많은 신문기자야."},
                {"role": "user", "content":"""다음과 같이 물가상승이 미래에 둔화될 것 같다는 뉴스 기사 30개를 작성해줘:
1. 전북지역 지난달 소비자물가가 2.9%올라, 20개월 만에 2%대로 떨어져 안정세를 되찾아가고 있다.
2. 석유류 가격이 1년 전보다 18% 떨어진 게 상승률 둔화의 주요 원인으로 작용했다.
3. 도시가스 요금은 지난해 인상폭(5.47원/MJ)에 크게 못 미칠 것으로 내다봤다.
..."""
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
    except Exception as e:
        print(f"Unexpected error: {e}")
        time.sleep(5.0)
        pass

#
# vim:sw=4:ts=4:expandtab:
# 
