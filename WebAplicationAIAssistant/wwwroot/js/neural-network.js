document.addEventListener('DOMContentLoaded', function () {
    const levelData = document.getElementById('level-data');
    const level = parseInt(levelData.getAttribute('data-level'));
    const inputCount = parseInt(levelData.getAttribute('data-input-count'));
    
    const sliders = document.querySelectorAll('.weight-slider');
    const valueDisplays = document.querySelectorAll('.weight-value');
    const testBtn = document.getElementById('test-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultsContainer = document.getElementById('results-container');

    // Обновление значений при изменении слайдеров
    sliders.forEach((slider, index) => {
        slider.addEventListener('input', function () {
            valueDisplays[index].textContent = parseFloat(this.value).toFixed(1);
        });
    });

    // Сброс всех весов
    resetBtn.addEventListener('click', function () {
        sliders.forEach((slider, index) => {
            slider.value = 0;
            valueDisplays[index].textContent = '0.0';
        });
        resultsContainer.innerHTML = '<div class="results-placeholder"><p>Измени веса и нажми "Проверить нейросеть" для тестирования</p></div>';
    });

    // Тестирование нейросети
    testBtn.addEventListener('click', async function () {
        const weights = Array.from(sliders).map(slider => parseFloat(slider.value));
        
        testBtn.disabled = true;
        testBtn.textContent = '⏳ Проверяю...';

        try {
            const response = await fetch('/NeuralNetwork/Test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    level: level,
                    weights: weights
                })
            });

            if (!response.ok) {
                throw new Error('Ошибка при тестировании');
            }

            const result = await response.json();
            displayResults(result);
        } catch (error) {
            console.error('Error:', error);
            resultsContainer.innerHTML = '<div class="error-message">Ошибка при проверке. Попробуй еще раз.</div>';
        } finally {
            testBtn.disabled = false;
            testBtn.textContent = '🧪 Проверить нейросеть';
        }
    });

    function displayResults(result) {
        let html = '';

        if (result.isCorrect) {
            html += `<div class="success-message">
                        <h4>🎉 ${result.message}</h4>
                        <p>Ты успешно обучил нейросеть! Все ответы правильные!</p>
                    </div>`;
        } else {
            html += `<div class="info-message">
                        <h4>📈 Результаты: ${result.correctCount} из ${result.totalCount} правильных</h4>
                        <p>${result.message}</p>
                    </div>`;
        }

        html += '<div class="results-table-container"><table class="results-table"><thead><tr>';
        html += '<th>№</th>';
        html += '<th>Входы</th>';
        html += '<th>S</th>';
        html += '<th>Ответ</th>';
        html += '<th>Ожидаемый</th>';
        html += '<th>Результат</th>';
        html += '</tr></thead><tbody>';

        result.results.forEach((example, index) => {
            const isCorrect = example.actualOutput === example.expectedOutput;
            const inputsStr = example.inputs.map(inp => inp === 1 ? 'Да' : 'Нет').join(', ');
            const answerStr = example.actualOutput === 1 ? 'Купить ✅' : 'Не покупать ❌';
            const expectedStr = example.expectedOutput === 1 ? 'Купить ✅' : 'Не покупать ❌';
            
            html += `<tr class="${isCorrect ? 'correct-row' : 'incorrect-row'}">`;
            html += `<td>${index + 1}</td>`;
            html += `<td>${inputsStr}</td>`;
            html += `<td>${example.sum.toFixed(2)}</td>`;
            html += `<td>${answerStr}</td>`;
            html += `<td>${expectedStr}</td>`;
            html += `<td>${isCorrect ? '✅ Правильно' : '❌ Неправильно'}</td>`;
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        resultsContainer.innerHTML = html;
    }
});
