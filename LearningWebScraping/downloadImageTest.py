import requests

with open('pic1.jpg', 'wb') as handle:
    response = requests.get("https://home.mindworking.eu/api/mediaData/mediaDataMediaPurposeId/SU1lZGlhRGF0YTp7Ik1lZGlhRGF0YUlkIjo1NzIyODc3LCJNZWRpYVB1cnBvc2VJZCI6MTR9/imageHash/bf0c009092f29a97d660e2b33b8cdc0d/imageSize/Assets/inline/True?deviceid=jd83hsdf3&width=1440", stream=True)

    if not response.ok:
        print(response)

    for block in response.iter_content(1024):
        if not block:
            break

        handle.write(block)