Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f41b4d4a1d0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.171525  0.118162  
    1      25.391064  0.242969  0.147052  
    2      24.304707  0.160455  0.118061  
    3      25.291103  0.031879  0.026856  
    4      25.096743  0.208165  0.156996  
    ...          ...       ...       ...  
    99995  24.737946  0.016516  0.009843  
    99996  24.224169  0.090827  0.048304  
    99997  25.613836  0.006648  0.006613  
    99998  25.274899  0.029639  0.026256  
    99999  25.699642  0.126614  0.065639  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.593304</td>
          <td>0.150058</td>
          <td>26.005740</td>
          <td>0.079247</td>
          <td>25.065448</td>
          <td>0.056262</td>
          <td>24.704518</td>
          <td>0.078228</td>
          <td>24.092030</td>
          <td>0.102453</td>
          <td>0.171525</td>
          <td>0.118162</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.914350</td>
          <td>0.515566</td>
          <td>27.197332</td>
          <td>0.249386</td>
          <td>26.366444</td>
          <td>0.108787</td>
          <td>26.165880</td>
          <td>0.147716</td>
          <td>25.844585</td>
          <td>0.209457</td>
          <td>25.035463</td>
          <td>0.229578</td>
          <td>0.242969</td>
          <td>0.147052</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.961112</td>
          <td>0.408452</td>
          <td>26.035704</td>
          <td>0.132035</td>
          <td>25.032685</td>
          <td>0.104389</td>
          <td>24.312535</td>
          <td>0.124163</td>
          <td>0.160455</td>
          <td>0.118061</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.174356</td>
          <td>0.621190</td>
          <td>28.720396</td>
          <td>0.778823</td>
          <td>27.393219</td>
          <td>0.260205</td>
          <td>26.310734</td>
          <td>0.167208</td>
          <td>25.598281</td>
          <td>0.170147</td>
          <td>25.334976</td>
          <td>0.293326</td>
          <td>0.031879</td>
          <td>0.026856</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.578368</td>
          <td>0.400572</td>
          <td>26.160603</td>
          <td>0.103143</td>
          <td>25.913909</td>
          <td>0.073071</td>
          <td>25.668610</td>
          <td>0.095879</td>
          <td>25.281777</td>
          <td>0.129659</td>
          <td>25.547942</td>
          <td>0.347603</td>
          <td>0.208165</td>
          <td>0.156996</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.741349</td>
          <td>0.904808</td>
          <td>26.472580</td>
          <td>0.135255</td>
          <td>25.420678</td>
          <td>0.047180</td>
          <td>25.123627</td>
          <td>0.059244</td>
          <td>24.777403</td>
          <td>0.083424</td>
          <td>24.439087</td>
          <td>0.138528</td>
          <td>0.016516</td>
          <td>0.009843</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.812148</td>
          <td>0.180823</td>
          <td>25.997955</td>
          <td>0.078704</td>
          <td>25.362623</td>
          <td>0.073217</td>
          <td>24.897680</td>
          <td>0.092738</td>
          <td>24.321254</td>
          <td>0.125105</td>
          <td>0.090827</td>
          <td>0.048304</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.136308</td>
          <td>0.604791</td>
          <td>26.583267</td>
          <td>0.148772</td>
          <td>26.340442</td>
          <td>0.106343</td>
          <td>26.365853</td>
          <td>0.175234</td>
          <td>25.702612</td>
          <td>0.185887</td>
          <td>25.627424</td>
          <td>0.369948</td>
          <td>0.006648</td>
          <td>0.006613</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.281702</td>
          <td>0.317508</td>
          <td>26.127797</td>
          <td>0.100226</td>
          <td>26.131708</td>
          <td>0.088552</td>
          <td>25.741792</td>
          <td>0.102231</td>
          <td>25.634860</td>
          <td>0.175520</td>
          <td>24.901893</td>
          <td>0.205387</td>
          <td>0.029639</td>
          <td>0.026256</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.875057</td>
          <td>0.190692</td>
          <td>26.483272</td>
          <td>0.120441</td>
          <td>26.110028</td>
          <td>0.140785</td>
          <td>26.302861</td>
          <td>0.305029</td>
          <td>25.632555</td>
          <td>0.371431</td>
          <td>0.126614</td>
          <td>0.065639</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.371079</td>
          <td>0.396364</td>
          <td>26.818878</td>
          <td>0.221156</td>
          <td>26.103695</td>
          <td>0.108685</td>
          <td>25.398180</td>
          <td>0.096021</td>
          <td>24.751845</td>
          <td>0.102636</td>
          <td>24.132171</td>
          <td>0.134153</td>
          <td>0.171525</td>
          <td>0.118162</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.750361</td>
          <td>0.481734</td>
          <td>26.413934</td>
          <td>0.149372</td>
          <td>25.997020</td>
          <td>0.169598</td>
          <td>26.015421</td>
          <td>0.312581</td>
          <td>25.741624</td>
          <td>0.516861</td>
          <td>0.242969</td>
          <td>0.147052</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.106859</td>
          <td>1.783151</td>
          <td>25.996397</td>
          <td>0.160407</td>
          <td>25.060396</td>
          <td>0.133526</td>
          <td>24.362478</td>
          <td>0.162628</td>
          <td>0.160455</td>
          <td>0.118061</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.133212</td>
          <td>0.663496</td>
          <td>30.364815</td>
          <td>2.041231</td>
          <td>27.245725</td>
          <td>0.268759</td>
          <td>26.315712</td>
          <td>0.198064</td>
          <td>25.734199</td>
          <td>0.223269</td>
          <td>24.393926</td>
          <td>0.157350</td>
          <td>0.031879</td>
          <td>0.026856</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.443525</td>
          <td>0.429124</td>
          <td>26.153751</td>
          <td>0.129650</td>
          <td>26.014020</td>
          <td>0.104043</td>
          <td>25.645458</td>
          <td>0.123469</td>
          <td>25.769424</td>
          <td>0.252525</td>
          <td>24.823269</td>
          <td>0.248906</td>
          <td>0.208165</td>
          <td>0.156996</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.690497</td>
          <td>0.952539</td>
          <td>26.540257</td>
          <td>0.164903</td>
          <td>25.384283</td>
          <td>0.053838</td>
          <td>25.169917</td>
          <td>0.073218</td>
          <td>24.865409</td>
          <td>0.105975</td>
          <td>24.945500</td>
          <td>0.249475</td>
          <td>0.016516</td>
          <td>0.009843</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.860809</td>
          <td>1.064362</td>
          <td>26.739899</td>
          <td>0.198157</td>
          <td>25.959462</td>
          <td>0.091069</td>
          <td>25.246034</td>
          <td>0.079706</td>
          <td>24.668235</td>
          <td>0.090682</td>
          <td>24.404889</td>
          <td>0.161175</td>
          <td>0.090827</td>
          <td>0.048304</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.445538</td>
          <td>0.816100</td>
          <td>26.618430</td>
          <td>0.176159</td>
          <td>26.670448</td>
          <td>0.165809</td>
          <td>26.391287</td>
          <td>0.210435</td>
          <td>25.510386</td>
          <td>0.184569</td>
          <td>25.234564</td>
          <td>0.315205</td>
          <td>0.006648</td>
          <td>0.006613</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.715749</td>
          <td>0.223751</td>
          <td>26.252909</td>
          <td>0.129079</td>
          <td>26.038445</td>
          <td>0.096160</td>
          <td>25.882670</td>
          <td>0.136892</td>
          <td>26.107068</td>
          <td>0.302781</td>
          <td>25.818420</td>
          <td>0.495331</td>
          <td>0.029639</td>
          <td>0.026256</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.494775</td>
          <td>0.425474</td>
          <td>26.812450</td>
          <td>0.213407</td>
          <td>26.641056</td>
          <td>0.166961</td>
          <td>26.537284</td>
          <td>0.245305</td>
          <td>26.228268</td>
          <td>0.342770</td>
          <td>25.728197</td>
          <td>0.475627</td>
          <td>0.126614</td>
          <td>0.065639</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>28.713652</td>
          <td>1.701702</td>
          <td>26.849691</td>
          <td>0.226071</td>
          <td>26.102464</td>
          <td>0.108080</td>
          <td>25.139356</td>
          <td>0.076090</td>
          <td>24.629739</td>
          <td>0.091803</td>
          <td>24.040849</td>
          <td>0.123379</td>
          <td>0.171525</td>
          <td>0.118162</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.531856</td>
          <td>1.649729</td>
          <td>27.950276</td>
          <td>0.588297</td>
          <td>26.666029</td>
          <td>0.198058</td>
          <td>26.168234</td>
          <td>0.209925</td>
          <td>26.601061</td>
          <td>0.520791</td>
          <td>25.220092</td>
          <td>0.370380</td>
          <td>0.242969</td>
          <td>0.147052</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.051478</td>
          <td>1.206376</td>
          <td>27.906904</td>
          <td>0.512959</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.025740</td>
          <td>0.162273</td>
          <td>24.925773</td>
          <td>0.117260</td>
          <td>24.104388</td>
          <td>0.128520</td>
          <td>0.160455</td>
          <td>0.118061</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.008183</td>
          <td>1.069316</td>
          <td>28.015500</td>
          <td>0.478639</td>
          <td>27.865433</td>
          <td>0.383292</td>
          <td>26.372413</td>
          <td>0.178307</td>
          <td>25.315698</td>
          <td>0.135059</td>
          <td>25.951447</td>
          <td>0.478670</td>
          <td>0.031879</td>
          <td>0.026856</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.299567</td>
          <td>0.399502</td>
          <td>26.147792</td>
          <td>0.135534</td>
          <td>25.967408</td>
          <td>0.105436</td>
          <td>25.721510</td>
          <td>0.139285</td>
          <td>25.228082</td>
          <td>0.169055</td>
          <td>26.480915</td>
          <td>0.883143</td>
          <td>0.208165</td>
          <td>0.156996</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.272881</td>
          <td>0.666084</td>
          <td>26.256923</td>
          <td>0.112429</td>
          <td>25.420239</td>
          <td>0.047281</td>
          <td>25.074554</td>
          <td>0.056870</td>
          <td>24.968014</td>
          <td>0.098888</td>
          <td>24.481355</td>
          <td>0.144032</td>
          <td>0.016516</td>
          <td>0.009843</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.823974</td>
          <td>0.501645</td>
          <td>26.818283</td>
          <td>0.192112</td>
          <td>25.988122</td>
          <td>0.083354</td>
          <td>25.227549</td>
          <td>0.069635</td>
          <td>24.643302</td>
          <td>0.079175</td>
          <td>24.334450</td>
          <td>0.135326</td>
          <td>0.090827</td>
          <td>0.048304</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.318648</td>
          <td>0.327098</td>
          <td>26.852201</td>
          <td>0.187143</td>
          <td>26.373093</td>
          <td>0.109485</td>
          <td>26.196197</td>
          <td>0.151704</td>
          <td>25.457236</td>
          <td>0.150916</td>
          <td>25.598202</td>
          <td>0.361796</td>
          <td>0.006648</td>
          <td>0.006613</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.414550</td>
          <td>0.355002</td>
          <td>26.358888</td>
          <td>0.123697</td>
          <td>26.157934</td>
          <td>0.091580</td>
          <td>26.399213</td>
          <td>0.182195</td>
          <td>26.113026</td>
          <td>0.264119</td>
          <td>25.035721</td>
          <td>0.232010</td>
          <td>0.029639</td>
          <td>0.026256</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.103550</td>
          <td>1.187323</td>
          <td>26.731007</td>
          <td>0.186559</td>
          <td>26.740647</td>
          <td>0.168630</td>
          <td>26.183598</td>
          <td>0.168959</td>
          <td>26.355375</td>
          <td>0.353553</td>
          <td>25.481834</td>
          <td>0.367776</td>
          <td>0.126614</td>
          <td>0.065639</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
