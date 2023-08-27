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
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content":"너는 유능하면서 경제 전문지식이 많은 신문기자야."},
                {"role": "user", "content":"""다음과 같은 식으로 인플레이션 심화에 직접적 또는 간접적 원인이 되는 기사 문장 31개를 생성해 줘. 단, 되도록이면  인플레이션이나 물가라는 단어를 직접적으로 언급하지는 말아줘:
1. KDI는 또 지난해의 높은 임금인상과 재정-통화-환율정책이 경제활력을 회복시키는 방향으로 운용될 계획이어서 물가상승 압력으로 나타날 것을 전망했다.
2. 일부 전문가는 미국 실업률 급감이 인플레이션을 급속히 끌어올릴 수 있다고 우려하고 있다.
3. 세계 공급망의 불안정성으로 수요에 공급이 따라가지를 못하고 있다.
4. 연방 준비 은행이 금리 인상을 예고하고 있다. 
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
    except:
        time.sleep(5.0)
        pass

#
# vim:sw=4:ts=4:expandtab:
# 
