// 로컬 스토리지에 예측 기록 저장
function savePredictionLocally(imageName, results) {
	const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
	history.push({
		imageName,
		results,
		timestamp: new Date().toISOString()
	});
	localStorage.setItem('predictionHistory', JSON.stringify(history));
}

// 저장된 예측 기록 불러오기 및 표시
function loadPredictionHistory() {
	const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
	const container = document.getElementById('history-list');
	if (!container) return;
	container.innerHTML = '';
	history.forEach(entry => {
		const div = document.createElement('div');
		div.innerHTML = `
		<strong>${entry.imageName}</strong> (${new Date(entry.timestamp).toLocaleString()})
		<ul>
		  ${entry.results.map(r => `<li>${r.className}: ${r.probability.toFixed(3)}</li>`).join('')}
		</ul>
		<hr/>
	  `;
		container.appendChild(div);
	});
}

// 이미지로부터 예측 실행
async function runPrediction(imageElement, filename = '업로드한 이미지') {
	if (!model) {
		console.error("❌ 모델이 아직 로드되지 않았습니다.");
		return;
	}

	const tensor = tf.browser.fromPixels(imageElement)
		.resizeNearestNeighbor([224, 224])
		.toFloat()
		.sub(tf.scalar(127.5))
		.div(tf.scalar(127.5))
		.expandDims();

	const predictions = await model.predict(tensor).data();
	const top3 = Array.from(predictions)
		.map((p, i) => ({
			probability: p,
			className: SKIN_CLASSES[i]
		}))
		.sort((a, b) => b.probability - a.probability)
		.slice(0, 3);

	const $list = $("#prediction-list");
	$list.empty();
	$list.append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${filename}</li>`);
	top3.forEach(p => {
		$list.append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(4)}</li>`);
	});

	savePredictionLocally(filename, top3);
}

// 업로드 이미지 처리
document.getElementById('uploadInput').addEventListener('change', function () {
	const file = this.files[0];
	if (!file) return;

	const reader = new FileReader();
	reader.onload = function (e) {
		const img = document.getElementById('selected-image');
		img.onload = () => runPrediction(img, file.name);
		img.src = e.target.result;
	};
	reader.readAsDataURL(file);
});

// 카메라 촬영 이미지 처리
document.getElementById('cameraInput').addEventListener('change', function () {
	const file = this.files[0];
	if (!file) return;

	const reader = new FileReader();
	reader.onload = function (e) {
		const img = document.getElementById('selected-image');
		img.onload = () => runPrediction(img, file.name);
		img.src = e.target.result;
	};
	reader.readAsDataURL(file);
});

// 기록 페이지 로드시 자동 표시
document.addEventListener('DOMContentLoaded', () => {
	loadPredictionHistory();
});
