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
        {"role": "system", "content":"너는 유능하면서 전문지식이 많은 신문기자야."},
        {"role": "user", "content":"""다음과 같은 식으로 물가와 관련이 없는 신문기사 문장 30개만 생성해줘. 반드시 경제와 관련된 기사일 필요는 없어:
1. 여객기 자체 생산을 추진중인 중국이 대형 여객기 첫 상업 비행에 성공한 것으로 전해졌다.
2. 이르면 오는 7월부터 운전자보험의 교통사고 처리지원금, 변호사 선임 비용 담보에 대해 자기 부담금을 최대 20%까지 추가할 예정이다.
3. 스타트업 기업들의 창업 생태계 발전을 위해 정부의 창업 지원 정책이 강화되고 있다.
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
