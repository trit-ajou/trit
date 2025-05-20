# 단어장
## TextedImage
- Pipeline을 따라 이동하는 핵심 데이터 클래스
### 멤버 변수
- .orig, .timg, .mask, .bboxes
- orig: 텍스트 없는 원본 이미지(torch.Tensor)
- timg: 텍스트 있는 이미지(torch.Tensor)
- mask: 픽셀 레벨 binary(float32) 마스크 이미지(torch.Tensor)
- bboxes: 텍스트 객체별 bounding box
### methods
- merge_bboxes_with_margin: self의 bbox 중 margin보다 가까운 것들을 merge
- split_margin_crop: 각 bbox 기준으로 margin crop하여 만든 TextedImage 리스트 반환. 모델2 입력으로 사용
- split_center_crop: 각 bbox 기준으로 center crop하여 만든 TextedImage 리스트 반환. 모델3 입력으로 사용
- merge_cropped: 위 두 split 함수의 역기능을 하는 함수
### crop 종류
- precise crop: bbox 영역을 정확히 crop. bbox.slice 사용
- margin crop: bbox를 margin만큼 확장한 다음 precise crop. 만약 확장한 bbox가 이미지 벗어나면 zero pad
- center crop: bbox를 중심으로 size만큼 crop. 만약 이미지 벗어나면 이미지 내부로 강제 평행이동

## BBox
- x1, y1, x2, y2 형태의 tuple

## ImageLoader
- 아무튼 생성됨 건드리지말것 어케돌아가는지 모름

## Utils/PipelineSetting, ImagePolicy
- 설정 클래스. 기본값 바꾸고싶으면 main에서 parser 기본값 수정하셈