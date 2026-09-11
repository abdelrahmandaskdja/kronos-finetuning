function chartLayout(overrides = {}) {
    return {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { t: 24, r: 18, b: 44, l: 54 },
        font: { family: "Space Grotesk, sans-serif", color: "#11222a" },
        xaxis: {
            gridcolor: "rgba(17,34,42,0.08)",
            zerolinecolor: "rgba(17,34,42,0.08)",
        },
        yaxis: {
            gridcolor: "rgba(17,34,42,0.08)",
            zerolinecolor: "rgba(17,34,42,0.08)",
        },
        legend: {
            orientation: "h",
            yanchor: "bottom",
            y: 1.02,
            xanchor: "left",
            x: 0,
        },
        ...overrides,
    };
}

function shortValue(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "n/a";
    }
    return Number(value).toLocaleString(undefined, {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits > 0 ? 0 : 0,
    });
}

function renderOverviewCharts(rows) {
    const labels = rows.map((row) => row.label);
    const accuracy = rows.map((row) => row.accuracy_pct);
    const baseline = rows.map((row) => row.baseline_pct);
    const pnl = rows.map((row) => row.pnl_usdt ?? 0);
    const pnlText = rows.map((row) => (row.pnl_usdt === null ? "n/a" : shortValue(row.pnl_usdt, 0)));

    Plotly.react("overview-accuracy-chart", [
        {
            x: labels,
            y: accuracy,
            name: "Model",
            type: "bar",
            marker: { color: "#0f8b8d" },
        },
        {
            x: labels,
            y: baseline,
            name: "Baseline",
            type: "bar",
            marker: { color: "#e76f51" },
        },
    ], chartLayout({
        barmode: "group",
        yaxis: { title: "Accuracy (%)", gridcolor: "rgba(17,34,42,0.08)" },
    }), { displayModeBar: false, responsive: true });

    Plotly.react("overview-pnl-chart", [
        {
            x: labels,
            y: pnl,
            text: pnlText,
            textposition: "outside",
            type: "bar",
            marker: {
                color: pnl.map((value) => (value >= 0 ? "#148f62" : "#cb4a5a")),
            },
        },
    ], chartLayout({
        yaxis: { title: "PnL (USDT)", gridcolor: "rgba(17,34,42,0.08)" },
    }), { displayModeBar: false, responsive: true });
}

function renderDetailCharts(payload) {
    const chart = payload.chart;
    const timestamps = chart.timestamps;
    const actualClose = chart.actual_close;
    const predictedClose = chart.predicted_close || [];
    const hasPredictedClose = predictedClose.some((value) => value !== null);

    const upX = [];
    const upY = [];
    const upColor = [];
    const downX = [];
    const downY = [];
    const downColor = [];

    chart.pred_dir.forEach((dir, index) => {
        const isCorrect = chart.correct[index];
        const color = isCorrect ? "#148f62" : "#cb4a5a";
        if (dir === 1) {
            upX.push(timestamps[index]);
            upY.push(chart.marker_y[index]);
            upColor.push(color);
        } else {
            downX.push(timestamps[index]);
            downY.push(chart.marker_y[index]);
            downColor.push(color);
        }
    });

    const priceTraces = [
        {
            x: timestamps,
            y: actualClose,
            type: "scatter",
            mode: "lines",
            name: "Actual close",
            line: { color: "#11222a", width: 2.4 },
        },
        {
            x: upX,
            y: upY,
            type: "scatter",
            mode: "markers",
            name: "Predicted up",
            marker: {
                color: upColor,
                size: 10,
                symbol: "triangle-up",
                line: { color: "#fff8ef", width: 1.2 },
            },
        },
        {
            x: downX,
            y: downY,
            type: "scatter",
            mode: "markers",
            name: "Predicted down",
            marker: {
                color: downColor,
                size: 10,
                symbol: "triangle-down",
                line: { color: "#fff8ef", width: 1.2 },
            },
        },
    ];

    if (hasPredictedClose) {
        priceTraces.splice(1, 0, {
            x: timestamps,
            y: predictedClose,
            type: "scatter",
            mode: "lines",
            name: "Predicted close",
            line: { color: "#e76f51", width: 1.8, dash: "dot" },
        });
    }

    Plotly.react("detail-price-chart", priceTraces, chartLayout({
        yaxis: { title: "Close price", gridcolor: "rgba(17,34,42,0.08)" },
    }), { displayModeBar: false, responsive: true });

    Plotly.react("detail-rolling-chart", [
        {
            x: payload.rolling_chart.timestamps,
            y: payload.rolling_chart.accuracy_pct,
            type: "scatter",
            mode: "lines",
            name: payload.rolling_chart.label,
            line: { color: "#0f8b8d", width: 2.4 },
        },
        {
            x: payload.rolling_chart.timestamps,
            y: payload.rolling_chart.timestamps.map(() => payload.rolling_chart.baseline_pct),
            type: "scatter",
            mode: "lines",
            name: "Baseline",
            line: { color: "#e76f51", width: 1.4, dash: "dash" },
        },
    ], chartLayout({
        yaxis: { title: "Accuracy (%)", gridcolor: "rgba(17,34,42,0.08)" },
    }), { displayModeBar: false, responsive: true });
}

function renderSparkline(card) {
    if (!card.market || !card.market.timestamps || !card.market.timestamps.length) {
        return;
    }
    const targetId = `sparkline-${card.slug}`;
    Plotly.react(targetId, [
        {
            x: card.market.timestamps,
            y: card.market.closes,
            type: "scatter",
            mode: "lines",
            line: { color: "#7bd6d2", width: 2 },
            fill: "tozeroy",
            fillcolor: "rgba(123,214,210,0.12)",
            hovertemplate: "%{x}<br>%{y}<extra></extra>",
        },
    ], {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { t: 6, r: 4, b: 18, l: 4 },
        xaxis: { visible: false },
        yaxis: { visible: false },
    }, { displayModeBar: false, responsive: true });
}

function renderForecastList(cardElement, card) {
    const target = cardElement.querySelector('[data-field="forecast-list"]');
    if (!target) {
        return;
    }
    if (card.forecast_rows && card.forecast_rows.length) {
        target.innerHTML = card.forecast_rows.map((row) => `
            <div class="forecast-mini-item">
                <span>${row.timestamp}</span>
                <strong>${row.direction} · ${shortValue(row.close)}</strong>
            </div>
        `).join("");
        return;
    }
    target.innerHTML = `
        <div class="forecast-mini-item">
            <span>No forecast rows</span>
            <strong>${card.signal.direction}</strong>
        </div>
    `;
}

function renderLiveCards(snapshot) {
    const generatedAt = document.getElementById("live-generated-at");
    if (generatedAt) {
        generatedAt.textContent = snapshot.generated_at;
    }

    snapshot.cards.forEach((card) => {
        const cardElement = document.querySelector(`[data-live-card="${card.slug}"]`);
        if (!cardElement) {
            return;
        }

        const badge = cardElement.querySelector(".signal-badge");
        badge.textContent = card.signal.direction;
        badge.classList.toggle("signal-up", card.signal.direction === "Up");
        badge.classList.toggle("signal-down", card.signal.direction === "Down");
        badge.classList.toggle("signal-neutral", card.signal.direction !== "Up" && card.signal.direction !== "Down");

        cardElement.querySelector('[data-field="signal-timestamp"]').textContent = card.signal.timestamp || "n/a";
        cardElement.querySelector('[data-field="market-close"]').textContent = card.market && card.market.last ? shortValue(card.market.last.close) : "n/a";
        cardElement.querySelector('[data-field="signal-move"]').textContent = card.signal.move_pct !== null && card.signal.move_pct !== undefined
            ? `${shortValue(card.signal.move_pct)}%`
            : "n/a";
        cardElement.querySelector('[data-field="signal-pred-close"]').textContent = card.signal.predicted_close !== null && card.signal.predicted_close !== undefined
            ? shortValue(card.signal.predicted_close)
            : "n/a";
        cardElement.querySelector('[data-field="market-lag"]').textContent = card.market_lag_minutes !== null && card.market_lag_minutes !== undefined
            ? `${shortValue(card.market_lag_minutes, 1)} min`
            : "n/a";
        cardElement.querySelector('[data-field="signal-lag"]').textContent = card.prediction_lag_minutes !== null && card.prediction_lag_minutes !== undefined
            ? `${shortValue(card.prediction_lag_minutes, 1)} min`
            : "n/a";
        const marketSource = cardElement.querySelector('[data-field="market-source"]');
        if (marketSource) {
            marketSource.textContent = `Market source: ${card.market_source_label || "n/a"}`;
        }
        cardElement.querySelector('[data-field="source-generated"]').textContent = `Inference refresh: ${card.source_generated_at || "n/a"}`;
        const errorTarget = cardElement.querySelector('[data-field="card-error"]');
        if (errorTarget) {
            errorTarget.textContent = card.error ? `Forecast error: ${card.error}` : "";
        }

        renderForecastList(cardElement, card);
        renderSparkline(card);
    });
}

async function pollLiveSnapshot() {
    try {
        const response = await fetch("/api/live-snapshot", { cache: "no-store" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        renderLiveCards(payload);
        const statusCopy = document.getElementById("live-status-copy");
        if (statusCopy) {
            statusCopy.textContent = "Polling active. Cards refreshed from local finetuned inference.";
        }
    } catch (error) {
        const statusCopy = document.getElementById("live-status-copy");
        if (statusCopy) {
            statusCopy.textContent = `Polling error: ${error.message}`;
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const page = document.body.dataset.page;

    if (page === "overview" && window.OVERVIEW_CHART_ROWS) {
        renderOverviewCharts(window.OVERVIEW_CHART_ROWS);
    }

    if (page === "detail" && window.DETAIL_PAYLOAD) {
        renderDetailCharts(window.DETAIL_PAYLOAD);
    }

    if (page === "live" && window.LIVE_SNAPSHOT) {
        renderLiveCards(window.LIVE_SNAPSHOT);
        const pollMs = Number(window.LIVE_SNAPSHOT.poll_seconds || 15) * 1000;
        window.setInterval(pollLiveSnapshot, pollMs);
    }
});
