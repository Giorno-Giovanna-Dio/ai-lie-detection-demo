# AI 微表情測謊分析 Demo

> 法庭科學課程專題：結合傳統測謊儀的生理原理，延伸至以 AI 分析受測者的臉部微表情。

**👉 線上 Demo：https://giorno-giovanna-dio.github.io/ai-lie-detection-demo/**

打開網址即可使用，不需要安裝任何東西。AI 會即時分析鏡頭或影片中的人物，並在畫面上顯示「說謊可能性指標」（例如：眉毛上揚、眼神飄移時，指標會升高到 80%）。所有運算都在瀏覽器內完成，影像不會上傳到任何伺服器。

---

## 課堂示範步驟

1. 用 Chrome 或 Safari 打開上面的網址，等右上角顯示「AI 模型就緒」。
2. 按 **開啟鏡頭**（允許鏡頭與麥克風），或 **上傳影片**（例如訪談、審訊、記者會片段）。
3. **前 10 秒是基準線**：請受測者自然回答中性問題（「你叫什麼名字？」「今天星期幾？」）。這對應到測謊儀的「控制問題」。
4. 基準線建立後開始提問。右上角儀表與右側面板會即時顯示指標、各項線索與情緒；異常時刻會記錄在「事件紀錄」。
5. 按 **● 錄製示範影片** 開始錄影，再按一次即停止並下載影片檔（含分析疊圖，可直接放進簡報）。

> 小遊戲建議：「兩真一假」：受測者講三件關於自己的事，其中一件是假的，讓全班看 AI 的指標猜哪一件是假的。

---

## 系統架構

```
影片 / 鏡頭
   │  ① 影像擷取（OpenCV）
   ▼
臉部特徵點：478 個 3D 點 + 52 個表情係數（MediaPipe Face Landmarker）
   │  ② 擷取行為線索：眉毛、眼神、眨眼、抿嘴、頭部、心率(rPPG)
   │  ③ 情緒分類：7 種基本情緒（DeepFace / face-api）
   ▼
④ 基準線：前 10 秒，計算每個線索的平均值與標準差
   ▼
⑤ 偏離程度 z = (目前值 − 基準平均) / 基準標準差
   指標 = sigmoid( Σ 權重ᵢ × max(0, zᵢ − 1) ) × 100%
```

| 專案提案中的技術 | 線上 Demo（瀏覽器版） | Python 參考版 [`python/analyze_video.py`](python/analyze_video.py) |
|---|---|---|
| OpenCV：讀取影像、畫疊圖 | HTML5 Video + Canvas | `cv2.VideoCapture`、`cv2.putText` |
| 臉部特徵點 | MediaPipe Face Landmarker（WebAssembly） | `mediapipe` |
| DeepFace：情緒辨識 | face-api 表情模型（同樣 7 類情緒） | `DeepFace.analyze(actions=["emotion"])` |

瀏覽器版用的是 OpenCV／DeepFace 的網頁等價工具，概念與步驟完全相同，只是能直接在網頁上執行。

### 偵測的線索

| 線索 | 怎麼量 | 權重 | 對應的生理／心理機制 |
|---|---|---|---|
| 眉毛上揚 | `browInnerUp`、`browOuterUp`（FACS AU1、AU2） | 1.0 | 驚訝、恐懼時額肌收縮 |
| 眼神飄移 | 虹膜方向偏離基準 | 1.0 | 認知負荷、迴避視線 |
| 眨眼頻率 | 每分鐘眨眼次數 | 0.6 | 交感神經喚起、認知負荷 |
| 抿嘴 | `mouthPress`、`mouthRoll`（AU24、AU28） | 0.7 | 壓抑情緒 |
| 頭部晃動 | 3D 頭部姿態角速度 | 0.5 | 緊張、坐立不安 |
| 驚訝／恐懼 | 情緒模型 fear + 0.7 × surprise | 0.8 | 害怕被拆穿 |
| 情緒突變 | 相鄰兩次情緒分布差異 | 0.5 | 微表情（0.04–0.2 秒） |
| 心率上升 | rPPG：額頭膚色隨脈搏的微小變化 | 0.6 | 自律神經反應，與測謊儀量測的心率相同 |

**與傳統測謊儀的對照**

| | 傳統測謊儀（Polygraph） | 本 AI Demo |
|---|---|---|
| 感測器 | 血壓袖帶、膚電、呼吸帶（需接觸） | 一般鏡頭（非接觸） |
| 量測 | 心率、血壓、汗腺（膚電）、呼吸 | 臉部肌肉動作、眼動、情緒、心率（rPPG） |
| 基準 | 控制問題（Control Question Test） | 前 10 秒基準線 |
| 本質 | 量測「生理喚起」而非「說謊」 | 同左：量測「偏離基準的程度」 |

---

## ⚠️ 科學限制（報告必須誠實說明）

- **沒有「說謊專屬」的表情或生理反應。** 這些線索反映的是緊張、認知負荷或情緒喚起，誠實但緊張的人也會出現。這與測謊儀的偽陽性問題相同（National Research Council, 2003）。
- 人類判斷說謊的平均準確率約 **54%**，只比亂猜好一點（Bond & DePaulo, 2006）。
- 微表情在說謊與誠實情境中都很少出現（Porter & ten Brinke, 2008）；微表情訓練工具也未能顯著提升判斷力（Jordan et al., 2019）。
- 情緒模型多以「擺拍」表情訓練，對真實、短暫的微表情辨識力有限；光線、角度、膚色、影片壓縮都會影響結果。
- 指標中的權重是依文獻方向設定的示範值，**未經資料集驗證**。下一步可用 Real-life Trial 資料集（Pérez-Rosas et al., 2015）實測準確率。
- 本工具僅供教學示範，**不得**用於真實偵查、審判、聘僱或任何對個人的判斷。

---

## 小組分工建議

| 科系 | 負責內容 |
|---|---|
| 資工、科技 | Demo 操作與技術說明、Python 版本、用資料集驗證準確率 |
| 生科 | 測謊儀的生理機制（自律神經、膚電、心率、呼吸）、臉部肌肉與 FACS、rPPG 原理 |
| 華語 | 文獻整理與報告撰寫；語言線索（含糊用語、細節多寡、CBCA／Reality Monitoring） |
| 競技 | 運動生理學：運動也會讓心率上升（偽陽性實例）；規劃並執行「兩真一假」實驗錄影 |

**國外案例方向**：EU iBorderCtrl（AI 邊境測謊試辦與爭議）、Converus EyeDetect（眼動測謊）、美國法院對測謊證據的態度（Frye 案, 1923）、台灣測謊證據在司法實務中的證據能力。

---

## 參考文獻（起點清單，請自行核對原文）

- Bond, C. F., & DePaulo, B. M. (2006). Accuracy of deception judgments. *Personality and Social Psychology Review, 10*(3), 214–234.
- DePaulo, B. M., et al. (2003). Cues to deception. *Psychological Bulletin, 129*(1), 74–118.
- Ekman, P., & Friesen, W. V. (1969). Nonverbal leakage and clues to deception. *Psychiatry, 32*(1), 88–106.
- Ekman, P. (2009). *Telling Lies: Clues to Deceit in the Marketplace, Politics, and Marriage* (rev. ed.). W. W. Norton.
- Jordan, S., et al. (2019). A test of the micro-expressions training tool: Does it improve lie detection? *Journal of Investigative Psychology and Offender Profiling, 16*(3).
- National Research Council. (2003). *The Polygraph and Lie Detection*. National Academies Press.
- Pérez-Rosas, V., Abouelenien, M., Mihalcea, R., & Burzo, M. (2015). Deception detection using real-life trial data. *Proceedings of ICMI 2015*.
- Porter, S., & ten Brinke, L. (2008). Reading between the lies: Identifying concealed and falsified emotions in universal facial expressions. *Psychological Science, 19*(5), 508–514.
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express, 16*(26), 21434–21445.
- Vrij, A. (2008). *Detecting Lies and Deceit: Pitfalls and Opportunities* (2nd ed.). Wiley.
- Wu, Z., Singh, B., Davis, L. S., & Subrahmanian, V. S. (2018). Deception detection in videos. *AAAI 2018*.
- Yan, W.-J., et al. (2014). CASME II: An improved spontaneous micro-expression database. *PLoS ONE, 9*(1), e86041.

**開源工具**：[OpenCV](https://github.com/opencv/opencv) · [DeepFace](https://github.com/serengil/deepface) · [MediaPipe](https://github.com/google-ai-edge/mediapipe) · [face-api (vladmandic)](https://github.com/vladmandic/face-api)

---

## 本機執行

```bash
python3 -m http.server 8000   # 然後打開 http://localhost:8000
```

Python 參考版說明見 [`python/analyze_video.py`](python/analyze_video.py) 檔頭（需要安裝 OpenCV、MediaPipe、DeepFace）。
