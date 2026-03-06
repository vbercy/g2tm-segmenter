# Results and Models

Below, we provide the results for different network settings and datasets. 

## ADE20K

We provide the evaluation results and the checkpoints of Segmenter models, that have been trained with G2TM applied at the 2nd layer with a varying threshold (curriculum) whose final value is 0.88.

<table>
    <tr>
        <th>Backbone</th>
        <th>Crop size</th>
        <th>mIoU</th>
        <th>Im/sec (BS=32)</th>
        <th>GFLOPs</th>
        <th colspan="3">Download</th>
    </tr>
    <tr>
        <td>ViT-Ti/16</td>
        <td>512x512</td>
        <td>40</td>
        <td>147.4</td>
        <td>12.8</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-Ti/16 + G2TM </td>
        <td>512x512</td>
        <td>39.9</td>
        <td>136.5</td>
        <td>7.8</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>512x512</td>
        <td>46.5</td>
        <td>94.9</td>
        <td>38.6</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16 + G2TM</td>
        <td>512x512</td>
        <td>46.6</td>
        <td>105.5</td>
        <td>26.1</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>512x512</td>
        <td>49.6</td>
        <td>43.2</td>
        <td>129.6</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16 + G2TM </td>
        <td>512x512</td>
        <td>48.1</td>
        <td>57.9</td>
        <td>82.5</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>512x512</td>
        <td>52.3</td>
        <td>15</td>
        <td>400.1</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16 + G2TM </td>
        <td>512x512</td>
        <td>50</td>
        <td>26</td>
        <td>211</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
</table>

## Cityscapes

We provide the evaluation results and the checkpoints of Segmenter models, that have been trained with G2TM applied at the 2nd layer with a varying threshold (curriculum) whose final value is 0.95.

### 768x768
<table>
    <tr>
        <th>Backbone</th>
        <th>Crop size</th>
        <th>mIoU</th>
        <th>Im/sec (BS=32)</th>
        <th>GFLOPs</th>
        <th colspan="3">Download</th>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>768x768</td>
        <td>73.5</td>
        <td>64.6</td>
        <td>43.5</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16 + G2TM</td>
        <td>768x768</td>
        <td>72</td>
        <td>72.7</td>
        <td>28.9</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>768x768</td>
        <td>76.6</td>
        <td>31.5</td>
        <td>116</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16 + G2TM </td>
        <td>768x768</td>
        <td>76.3</td>
        <td>46.2</td>
        <td>84</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>768x768</td>
        <td>77.6</td>
        <td>14.4</td>
        <td>348</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16 + G2TM</td>
        <td>768x768</td>
        <td>76</td>
        <td>21.8</td>
        <td>230</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>768x768</td>
        <td>79.1</td>
        <td>5.27</td>
        <td>1045</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16 + G2TM</td>
        <td>768x768</td>
        <td>76.8</td>
        <td>9.76</td>
        <td>599</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
</table>

### 1024x1024
<table>
    <tr>
        <th>Backbone</th>
        <th>Crop size</th>
        <th>mIoU</th>
        <th>Im/sec (BS=32)</th>
        <th>GFLOPs</th>
        <th colspan="3">Download</th>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>768x768</td>
        <td>73.1</td>
        <td>32.8</td>
        <td>117</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16 + G2TM</td>
        <td>768x768</td>
        <td>71.9</td>
        <td>38</td>
        <td>70.9</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>768x768</td>
        <td>76.6</td>
        <td>15.7</td>
        <td>285</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16 + G2TM </td>
        <td>768x768</td>
        <td>75.6</td>
        <td>20.9</td>
        <td>174</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>768x768</td>
        <td>77.1</td>
        <td>6.79</td>
        <td>776</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16 + G2TM</td>
        <td>768x768</td>
        <td>75.2</td>
        <td>11.2</td>
        <td>439</td>
        <td><a href="#results-and-models">model</a></td>
        <td><a href="#results-and-models">config</a></td>
        <td><a href="#results-and-models">log</a></td>
    </tr>
</table>

## Pascal Context
TBP
