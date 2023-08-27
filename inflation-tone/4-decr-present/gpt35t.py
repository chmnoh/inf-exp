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
                {"role": "user", "content":"""다음과 같은 식으로 인플레이션이 완화되고 있다는 직접적 또는 간접적 증거가 되는 기사 문장 31개를 생성해 줘. 단, 되도록이면  인플레이션이나 물가라는 단어를 직접적으로 언급하지는 말아줘::
1. 석유류 가격이 1년 전보다 18% 떨어진 게 상승률 둔화의 주요 원인으로 작용했다.
2. 농축산물 소비자물가 1.4% 하락, 공급여건 개선 안정세 전망이다.
3. 국제 유가가 한 달간 큰 폭으로 떨어지고 달러는 다시 강세로 돌아선 것이다.
4. 재정정책의 효과로 국내인들의 소비성향이 변화하여 목요일에 실시된 토종 과일의 물가가 하락세를 보이고 있다.
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
