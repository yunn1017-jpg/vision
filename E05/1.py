import numpy as np  # 배열 연산을 위해 NumPy를 불러옵니다.
import tensorflow as tf  # 딥러닝 모델 구성을 위해 TensorFlow를 불러옵니다.

tf.random.set_seed(42)  # 학습 재현성을 위해 TensorFlow 시드를 고정합니다.
np.random.seed(42)  # 학습 재현성을 위해 NumPy 시드를 고정합니다.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # MNIST 학습/테스트 데이터를 로드합니다.

x_train = x_train.astype("float32") / 255.0  # 학습 이미지를 float32로 바꾸고 0~1 범위로 정규화합니다.
x_test = x_test.astype("float32") / 255.0  # 테스트 이미지를 float32로 바꾸고 0~1 범위로 정규화합니다.

x_train = x_train.reshape(-1, 28 * 28)  # 학습 이미지를 28x28에서 784차원 벡터로 펼칩니다.
x_test = x_test.reshape(-1, 28 * 28)  # 테스트 이미지를 28x28에서 784차원 벡터로 펼칩니다.

model = tf.keras.Sequential(  # 순차적으로 레이어를 쌓는 모델을 생성합니다.
    [  # 모델 레이어 목록을 정의합니다.
        tf.keras.layers.Input(shape=(784,)),  # 입력 벡터 크기를 784로 지정합니다.
        tf.keras.layers.Dense(256, activation="relu"),  # 첫 번째 은닉층을 ReLU 활성화로 구성합니다.
        tf.keras.layers.Dense(128, activation="relu"),  # 두 번째 은닉층을 ReLU 활성화로 구성합니다.
        tf.keras.layers.Dense(10, activation="softmax"),  # 10개 숫자 클래스를 위한 출력층을 구성합니다.
    ]  # 레이어 목록 정의를 종료합니다.
)  # 모델 생성을 완료합니다.

model.compile(  # 학습을 위한 손실함수와 최적화기를 설정합니다.
    optimizer="adam",  # Adam 최적화기를 사용합니다.
    loss="sparse_categorical_crossentropy",  # 정수 라벨에 맞는 다중분류 손실을 사용합니다.
    metrics=["accuracy"],  # 평가 지표로 정확도를 사용합니다.
)  # 컴파일 설정을 완료합니다.

history = model.fit(  # 모델 학습을 시작합니다.
    x_train,  # 학습 입력 데이터를 전달합니다.
    y_train,  # 학습 정답 라벨을 전달합니다.
    epochs=5,  # 전체 데이터를 5회 반복 학습합니다.
    batch_size=128,  # 배치 크기를 128로 설정합니다.
    validation_split=0.1,  # 학습 데이터의 10%를 검증용으로 분리합니다.
    verbose=1,  # 학습 로그를 출력합니다.
)  # 학습 수행을 완료하고 이력을 저장합니다.

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)  # 테스트셋에서 손실과 정확도를 계산합니다.

print(f"MNIST Test Loss: {test_loss:.4f}")  # 테스트 손실을 소수점 4자리로 출력합니다.
print(f"MNIST Test Accuracy: {test_acc:.4f}")  # 테스트 정확도를 소수점 4자리로 출력합니다.

sample_indices = np.arange(10)  # 예시로 확인할 테스트 인덱스 0~9를 생성합니다.
sample_images = x_test[sample_indices]  # 선택한 인덱스의 테스트 이미지를 가져옵니다.
sample_labels = y_test[sample_indices]  # 선택한 인덱스의 정답 라벨을 가져옵니다.

pred_probs = model.predict(sample_images, verbose=0)  # 예시 이미지에 대한 클래스 확률을 예측합니다.
pred_labels = np.argmax(pred_probs, axis=1)  # 확률이 가장 큰 클래스를 최종 예측값으로 변환합니다.

for idx, true_label, pred_label in zip(sample_indices, sample_labels, pred_labels):  # 인덱스, 정답, 예측을 순회합니다.
    print(f"index={idx:02d} | true={true_label} | pred={pred_label}")  # 각 샘플의 정답과 예측을 출력합니다.