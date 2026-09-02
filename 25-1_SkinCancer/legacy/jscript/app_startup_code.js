function simulateClick(tabID) {
    document.getElementById(tabID).click();
}

function predictOnLoad() {
    setTimeout(simulateClick.bind(null, 'predict-button'), 500);
}

let model;
(async function () {
    model = await tf.loadLayersModel('final_model_kaggle_version/model.json');
    $("#selected-image").attr("src", "assets/samplepic.jpg");
    $('.progress-bar').hide();
    predictOnLoad();
})();

$("#predict-button").click(async function () {
    // 이미지 전처리 및 예측
    let image = $('#selected-image').get(0);
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat();
    let offset = tf.scalar(127.5);
    tensor = tensor.sub(offset).div(offset).expandDims();

    let predictions = await model.predict(tensor).data();
    let top3 = Array.from(predictions)
        .map((p, i) => ({ probability: p, className: SKIN_CLASSES[i] }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3);

    // 업로드된 파일명 가져오기
    const up = document.getElementById('upload-image-selector');
    const cam = document.getElementById('camera-image-selector');
    const file = up.files[0] || cam.files[0] || null;
    const fileName = file ? file.name : 'samplepic.jpg';

    // 결과 표시
    $("#prediction-list").empty();
    $("#prediction-list").append(
        `<li class="w3-text-blue fname-font" style="list-style-type:none;">${fileName}</li>`
    );
    top3.forEach(p => {
        $("#prediction-list").append(
            `<li style="list-style-type:none;">${p.className}: ${(p.probability * 100).toFixed(2)}%</li>`
        );
    });

    // Firebase 저장
    if (file) {
        const ref = firebase.storage().ref(`images/${Date.now()}_${fileName}`);
        await ref.put(file);
        const imageUrl = await ref.getDownloadURL();

        await firebase.firestore().collection('results').add({
            imageUrl,
            fileName,
            predictions: top3,
            timestamp: firebase.firestore.FieldValue.serverTimestamp()
        });
    }
});

$("#image-selector").change(async function () {
    const fileList = $("#image-selector").prop('files');
    model_processArray(fileList);
});
