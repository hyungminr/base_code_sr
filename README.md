# Base Code for Single Image Super Resolution

## Benchmark Code:
  - ### Method
    - **RCAN**
    - Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
    - https://github.com/yulunzhang/RCAN 
  - ### Dataset
    - **DIV2K**
    - LR Interpolation = bicubic
    - scale_facor = 2
    - batch size = 4
    - train image size = 3 x 128 x 128 (1 ~ 800, Random Cropped)
    - test image size = 3 x 256 x 256 (801 ~ 900, Center Cropped)
 
## Benchmark Result:

| Train Epoch | PSNR | SSIM |
| -------:| :-----: | :-----: |
|10|29.2049|0.9088|
|20|29.0079|0.9000|
|30|30.0547|0.9176|
|40|30.6276|0.9240|
|50|30.2934|0.9185|
|60|30.5399|0.9222|
|70|30.4420|0.9217|
|80|30.6044|0.9232|
|90|30.8959|0.9261|
|100|30.7759|0.9260|
|110|30.2820|0.9207|
|120|30.9784|0.9270|
|130|31.0422|0.9277|
|140|31.0701|0.9281|
|150|31.0822|0.9281|
|160|30.9238|0.9265|
|170|31.0863|0.9284|
|180|31.1980|0.9290|
|190|31.2526|0.9313|
|200|31.3149|0.9304|
|210|31.3972|0.9320|
|220|31.3125|0.9313|
|230|31.2126|0.9291|
|240|31.4179|0.9312|
|250|31.3049|0.9300|
|260|31.3508|0.9308|
|270|31.2130|0.9304|
|280|31.4768|0.9322|
|290|31.5695|0.9330|
|300|31.5739|0.9334|
|310|31.3750|0.9305|
|320|31.4188|0.9315|
|330|31.6311|0.9338|
|340|31.4650|0.9317|
|350|31.4256|0.9308|
|360|31.3955|0.9312|
|370|31.6262|0.9334|
|380|31.4818|0.9319|
|390|31.4110|0.9308|
|400|31.5444|0.9326|
|410|31.5235|0.9321|
|420|31.3787|0.9305|
|430|31.4964|0.9316|
|440|31.4331|0.9311|
|450|31.5512|0.9321|
|460|31.4363|0.9315|
|470|31.4339|0.9305|
|480|31.4623|0.9312|
|490|31.5052|0.9314|
|500|31.5133|0.9321|
|510|31.4936|0.9320|
|520|31.7493|0.9345|
|530|31.5725|0.9321|
|540|31.5608|0.9324|
|550|31.5949|0.9326|
|560|31.6903|0.9340|
|570|31.4466|0.9312|
|580|31.7145|0.9337|
|590|31.6790|0.9337|
|600|31.6325|0.9334|
|610|31.5760|0.9326|
|620|31.6156|0.9330|
|630|31.7457|0.9344|
|640|31.6370|0.9334|
|650|31.5912|0.9330|
|660|31.4901|0.9313|
|670|31.7765|0.9347|
|680|31.5703|0.9326|
|690|31.6183|0.9333|
|700|31.6264|0.9331|
|710|31.5461|0.9319|
|720|31.5799|0.9324|
|730|31.6821|0.9341|
|740|31.6534|0.9334|
|750|31.4868|0.9315|
|760|31.6584|0.9332|
|770|31.5668|0.9325|
|780|31.7001|0.9339|
|790|31.7423|0.9341|
|800|31.6481|0.9330|
|810|31.8311|0.9354|
|820|31.6730|0.9336|
|830|31.7060|0.9339|
|840|31.6881|0.9333|
|850|31.7332|0.9341|
|860|31.6453|0.9328|
|870|31.6350|0.9330|
|880|31.6017|0.9327|
|890|31.7984|0.9349|
|900|31.7615|0.9344|
|910|31.8031|0.9351|
|920|31.7341|0.9341|
|930|31.6782|0.9335|
|940|31.7217|0.9341|
|950|31.7902|0.9350|
|960|31.7104|0.9339|
|970|31.7778|0.9344|
|980|31.7918|0.9348|
|990|31.7269|0.9340|
|1000|31.7962|0.9349|
