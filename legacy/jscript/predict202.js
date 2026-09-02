function simulateClick(tabID) {
	document.getElementById(tabID).click();
}

function predictOnLoad() {
	setTimeout(simulateClick.bind(null, 'predict-button'), 500);
}

$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	};

	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
	setTimeout(simulateClick.bind(null, 'predict-button'), 500);
});

let skinFilterModel, lesionModel;

(async function () {
    skinFilterModel = await tf.loadLayersModel('final_model_kaggle_version/skin_filter_model/model.json');
    lesionModel = await tf.loadLayersModel('final_model_kaggle_version/lesion_model/model.json');
    $("#selected-image").attr("src", "assets/samplepic.jpg");
    $('.progress-bar').hide();
    predictOnLoad();
})();


$("#predict-button").click(async function () {
    let image = $('#selected-image').get(0);
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat();
    let offset = tf.scalar(127.5);
    tensor = tensor.sub(offset).div(offset).expandDims();

    // 1. 먼저 사전 필터 모델로 피부 여부 판단
    const skinProb = await skinFilterModel.predict(tensor).data();

    $("#prediction-list").empty();
    if (skinProb[0] < 0.5) {
        $("#prediction-list").append(`<li>❌ 이 이미지는 피부 이미지가 아닙니다.</li>`);
        return;
    }

    // 2. 병변 모델 실행
    const predictions = await lesionModel.predict(tensor).data();
    const top3 = Array.from(predictions)
        .map((p, i) => ({
            probability: p,
            className: SKIN_CLASSES[i]
        }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3);

    top3.forEach(p => {
        $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
    });
});

