from openai import OpenAI

client = OpenAI(api_key='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')

def generate_reviews(prompt, count=1):
    reviews = []
    for i in range(count):
        review_generated = False
        while not review_generated:
            try:
                # Generate a response using the ChatCompletion method
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "system", "content": "You are a fast high quality covid|general health|mental health|breast cancer as well as Non comminicable diseases mainly for the african region but not limited to africa text report generating assistant. This text will be used to validatea a model. Your response will be accurate based off real world data you were trained on. DO not hallucinate as it will impact the quality of the data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    top_p=1,
                    frequency_penalty=2,
                    presence_penalty=2
                )
                review = response.choices[0].message.content.strip()
                word_count = len(review.split())
                print("word count:", word_count)
                print("counted")
                reviews.append(review)
                print(review)  # Add this line to print the generated review
                review_generated = True

                # Check if the word count is within the desired range
                if 15 <= word_count <= 70:
                    print("counted")
                    reviews.append(review)
                    review_generated = True

            except Exception as err:
                print(f"Encountered an error: {err}")

        # Optional: Add a slight variation to the prompt for next iteration
        prompt += " Provide another perspective."

    return reviews

# Corrected call to the function with a COVID-19 related prompt
generated_reviews = generate_reviews("give a high-quality full validation text report on COVID-19. for a model being trained on covid data\
    also give general health reports outside covid\
        - mental health \
        - cpmmon african diseases \
        - breast cancer \
        - Non comminicable diseases... etc \
        - do it around africa but not limited to africa", count=100)

# Iterate over the generated reviews
for idx, review in enumerate(generated_reviews):
    print(f"Review {idx + 1}: {review}")