import asyncio
from openai import AsyncOpenAI

api_key = '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'
client = AsyncOpenAI(api_key=api_key)

async def generate_review(prompt):
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a fast high quality covid | general health | mental health | breast cancer as well as Non comminicable diseases \
                    mainly for the african region but not limited to africa text report generating assistant. \
                    This text will be used to validatea a model. Your response will be accurate based off real world data you were trained on. \
                    DO not hallucinate as it will impact the quality of the data. \
                    You will not concatenate words."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=1,
            frequency_penalty=2,
            presence_penalty=2
        )
        return response.choices[0].message.content.strip()
    except Exception as err:
        print(f"Encountered an error: {err}")
        return None

async def main():
    prompt = "give a high-quality full validation text report on COVID-19. for a model being trained on Covid data \
        also give general health reports outside Covid\
            - mental health \
            - common African diseases \
            - breast cancer \
            - Non communicable diseases... etc \
            - do it around Africa but not limited to Africa \
            Do not concatenate words"
    tasks = [generate_review(prompt) for _ in range(200)]
    reviews = await asyncio.gather(*tasks)

    for idx, review in enumerate(reviews):
        if review is not None:
            print(f'{review}')


asyncio.run(main())