import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import fbeta_score

# 모델 생성
def create_model(img_dim=(128, 128, 3)):
    input_tensor = Input(shape=img_dim)
    # input layer 설정
    base_model = VGG16(include_top=False,
                       weights='imagenet',
                       input_shape=img_dim)
    # keras를 통한 vgg16 설정
    bn = BatchNormalization()(input_tensor)
    # batch normalization 수행
    x = base_model(bn)
    x = Flatten()(x)
    # input을 flatten
    output = Dense(17, activation='sigmoid')(x)
    # output은 17개 labels 분류할 수 있게 18개로 설정
    model = Model(input_tensor, output)
    return model

def predict(model, preprocessor, batch_size=128):
    # test 데이터에 대해서 예측 수행
    
    generator = preprocessor.get_prediction_generator(batch_size)
    #batch 크기로 예측 수행 모델 생성
    predictions_labels = model.predict_generator(generator=generator, verbose=1,
                                                 steps=len(preprocessor.X_test_filename) / batch_size)
    assert len(predictions_labels) == len(preprocessor.X_test), \ # 예측한 label과 정답 label의 길이가 같으면 각 길이를 출력
        "len(predictions_labels) = {}, len(preprocessor.X_test) = {}".format(
            len(predictions_labels), len(preprocessor.X_test))
    return predictions_labels, np.array(preprocessor.X_test)

# multi hot encoding 된 결과를 label에 매핑, 단 threshold 보다 클 경우에만
def map_predictions(preprocessor, predictions, thresholds):
        predictions_labels = []
        for prediction in predictions:
            labels = [preprocessor.y_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels
# 측정방법을 f_beta score로 사용
def fbeta(model, X_valid, y_valid):
    p_valid = model.predict(X_valid)
    return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')
