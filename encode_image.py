import base64

# 1. 여기에 변환할 이미지 파일의 경로를 입력하세요.
# 예: 'fairy.gif', 'images/my_icon.png'
image_path = 'fairy.gif'

try:
    with open(image_path, "rb") as image_file:
        # 파일을 읽고 Base64로 인코딩합니다.
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # 웹페이지(HTML)에서 바로 사용할 수 있는 data URI 형식으로 출력합니다.
        # 이 출력값을 복사해서 <img> 태그의 src 속성에 붙여넣으세요.
        print(f"data:image/gif;base64,{encoded_string}")

except FileNotFoundError:
    print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
    print("image_path 변수에 정확한 파일 경로를 입력했는지 확인해주세요.")