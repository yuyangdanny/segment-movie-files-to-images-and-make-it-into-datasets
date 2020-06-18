# segment-movie-files-to-images-and-make-it-into-datasets
自動切割影片為圖像作為資料集-以手部姿態估計為例:
<br>原理:將手部辨識的凍結模型導入Addpose.py後，再透過webcam檢測到的bounding-box內部圖片一偵一偵的擷取下來
<br>　　並統一存到一個資料夾中，這樣就不需要額外用label工具額外繪製標註圖。
<br>
<br>手比一的圖像經過real-time手部辨識過後切割成的圖像
<br>![手比一的圖像](https://github.com/yuyangdanny/segment-movie-files-to-images-and-make-it-into-datasets/blob/master/one_1_1651.png)
<br>手比二的圖像經過real-time手部辨識過後切割成的圖像
<br>![手比二的圖像](https://github.com/yuyangdanny/segment-movie-files-to-images-and-make-it-into-datasets/blob/master/two_1_794.png)
<br>
<br>*問題:手不辨識完後切割的圖片過小及有些許模糊，會繼續修正優化。
<br>該切割影片程式碼在A.py檔已有詳細註解。

