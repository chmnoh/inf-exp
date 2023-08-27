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
                {"role": "user", "content":"""다음과 같은 식으로 물가상승이 심화됐다는 뉴스 기사 30개를 작성해줘:
1. 공공요금인 전기, 가스, 수도요금은 폭등했고, 여기에다 외식물가 역시 높은 상승세를 지속하면서 서민들의 가게부담은 여전히 가중되고 있는 것으로 분석됐다.
2. 한국소비자원 '참가격'에 따르면 소비자들이 많이 찾는 8개 외식 품목의 지난달 서울지역 평균 가격이 작년보다 최고 13% 가까이 뛰었다.
3. 돼지고기는 수요가 늘어난 반면, 유럽산 수입단가가 오르면서 공급이 감소했다.
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
