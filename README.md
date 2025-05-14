안녕하세요! 만화 식작업을 도와주는 AI 서비스를 만들고자 합니다.
만화 이미지에서 텍스트 영역을 인식하여 지우고 inpainting하는 단계를 거칩니다.
다른 서비스와의 차별점은, 텍스트 영역을 bounding box로 인식하지 않고 픽셀 단위 마스크를 생성해내어 지운다는 점입니다. 이를 통해 정보 손실을 최소화하고 결과적으로 inpainting 품질을 올릴 수 있을 것으로 예상합니다.

이 서비스를 구현하기 위해 다음과 같은 모델이 필요합니다:

# 필요 모델
1. 텍스트 객체 bbox 검출 모델(이하 모델1)
2. 픽셀 단위 마스크 생성 모델(이하 모델2)
3. mask가 있는 inpainting 모델(이하 모델3)

전체 서비스 파이프라인은 다음과 같습니다:

# 용어 설명
orig: 텍스트 없는 이미지
timg: 텍스트 있는 이미지
mask: 마스크 이미지(ground truth). 모델2 출력 사용 설정 시 대체될 수 있음
bboxes: 텍스트 객체의 bbox 리스트(ground truth). 모델1 출력 사용 설정 시 대체될 수 있음

margin crop: 수 픽셀의 마진을 두고 bbox 영역을 crop 합니다.
precise crop: margin 없이 crop 합니다. margin=0 인 margin crop과 동일.
center crop: bbox를 중앙으로 하고 모델의 input size만큼 crop 합니다.

* 모델 1과 모델 2의 출력은 설정에 따라 기존의 데이터를 대체할 수도, 안 할 수도 있습니다.
* 모든 resize는 종횡비를 유지하므로 pad와 함께 사용합니다.

# 데이터 생성 파이프라인
1
orig, timg, mask, bboxes 생성 및 orig, timg, mask를 GPU(torch.Tensor)로 이동. 이하 모든 이미지 처리는 GPU에서 동작합니다.

2
각 bbox를 margin만큼 확장한 영역이 겹친다면 병합: 모델2의 입력에 다른 텍스트 조각이 들어가지 않도록.
만약 모델2의 입력보다 bbox가 더 커지더라도 resize 하기 때문에 괜찮음. 품질 저하가 심하다면 차후 로직 변경.

3
모델1 입력: timg를 resize, pad
모델1 타겟: bboxes

4
모델1 출력 후처리: 모델 1의 출력을 bbox list로 후처리합니다.

5
모델1 출력을 사용한다면: bboxes 대체
만약 모델1 정확도가 떨어진다면 이후 모델에서 mask map의 엉뚱한 부분을 crop하게 됨. 모델 1이 충분히 학습되기 전 사용하지 말 것.

6
모델2 입력: timg를 margin crop, resize, pad
모델2 타겟: mask를 precise crop, pad(margin), resize, pad

7
모델2 출력을 사용한다면: bbox별로 생성한 마스크를 union하여 mask 대체

8-9: 각 bbox에 대해 반복

8
모델3 입력1: timg center crop
모델3 입력2: mask precise crop, pad
모델3 타겟: timg에 'orig에서 bbox 영역을 precise crop한 것'을 덮어씌운 것

9
timg = 모델3 output 대체 

파이프라인은 여기까지입니다. 아래는 클래스별 설명입니다.

# 클래스별 설명
* BBox(tuple 상속)
x1, y1, x2, y2 를 포함합니다. 병합 로직을 포함합니다. parent Textedimage 객체에 대한 참조를 갖고 있어서, 해당 Bbox가 어떤 TextedImage의 것인지 조회할 수 있습니다.

* ImageLoader
설정한 디렉토리에서 이미지를 로딩하거나, 가우시안 노이즈 이미지를 생성합니다.
텍스트 추가 policy를 관리하고 bbox 계산을 담당합니다.
orig, timg, mask, bboxes로 구성된 데이터셋을 리턴하는 메서드가 필요합니다.

* ImagePolicy
ImageLoader에서 텍스트 삽입 policy를 저장합니다.

* TextedImage
파이프라인을 따라 이동할 핵심 데이터 container입니다.
orig, timg, mask, bboxes 를 포함합니다.

* PipelineMgr
전체 파이프라인을 관리합니다. 각 데이터셋에 TextedImage list를 공급하고, 설정에 따라 모델 1의 출력과 모델 2의 출력을 TextedImage에 덮어씌우는 역할을 합니다.

* PipelineSetting
Pipeline에 필요한 설정을 저장합니다.

* MangaDataset1
모델 1을 위한 데이터셋입니다.
__getitem__ 메서드에서, TextedImage list로부터 모델 1을 위한 입력과 타겟을 생성합니다.

* MangaDataset2
모델 2를 위한 데이터셋입니다.
__getitem__ 메서드에서, TextedImage list로부터 모델 2를 위한 입력과 타겟을 생성합니다.
각 TextedImage마다 여러 개의 Bbox가 있으므로, length는 모든 TextedImage의 BBox 개수의 합이 됩니다.

* MangaDataset3
모델 3을 위한 데이터셋입니다.
__getitem__ 메서드에서, TextedImage list로부터 모델 2를 위한 입력과 타겟을 생성합니다.
각 TextedImage마다 여러 개의 Bbox가 있으므로, length는 모든 TextedImage의 BBox 개수의 합이 됩니다.


# 구현 세부사항
전체 파이프라인과 TextedImage 의 data flow에 집중합니다.
ImageLoader의 policy과 bbox 계산은 dummy로 남겨두십시오.
각 모델 아키텍처와 모델1 후처리 함수는 dummy로 남겨두십시오.

모델 1만을 추론시킬 수도 있고, 모델 2만을 추론시킬 수도 있고, 모델 3만을 추론시킬 수 있어야 합니다. 동시에 여러 모델을 추론시킬 수 있어야 합니다.
모델 1만을 학습시킬 수도 있고, 모델 2만을 학습시킬 수도 있고, 모델 3만을 학습시킬 수도 있어야 합니다. 동시에 여러 모델을 학습시킬 수 있어야 합니다.
모델 2를 학습 시킬 때 모델 1의 출력을 사용할 수도 안 할 수도 있어야 합니다.
모델 3을 학습 시킬 때 모델 1, 2의 출력을 사용할 수도, 안 할 수도, 모델 1의 출력을 사용하고 모델 2의 출력을 사용하지 않을 수도, 모델 1의 출력을 사용하지 않고 모델 2의 출력을 사용할 수도 있어야 합니다.

파이프라인의 종료 직전에 각 데이터셋에서 샘플 하나씩을 시각화합니다(data1.png, data2.png, data3.png).

각 기능은 다음과 같은 Commandline interface를 통해 조절할 수 있어야 합니다:

--model1 {train,inference}
--model2 {train,inference}
--model3 {train,inference}

각 플래그가 설정되어있지 않다면, 해당 모델을 스킵하고 합성 데이터가 그대로 다음 파이프라인으로 넘어갑니다.
각 플래그가 train이라면, 해당 모델을 학습시키되 합성 데이터가 그대로 다음 파이프라인으로 넘어갑니다.
각 플래그가 inference라면, 해당 모델을 추론시키고 합성 데이터를 덮어씌웁니다.
모든 플래그가 설정되어있지 않다면, 데이터셋 시각화만 하고 프로그램을 종료합니다.
