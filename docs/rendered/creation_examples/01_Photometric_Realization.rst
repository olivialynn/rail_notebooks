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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f5b8d56c400>



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
    0      23.994413  0.179137  0.150946  
    1      25.391064  0.052724  0.051298  
    2      24.304707  0.058445  0.053692  
    3      25.291103  0.065644  0.056715  
    4      25.096743  0.043958  0.022187  
    ...          ...       ...       ...  
    99995  24.737946  0.004769  0.003336  
    99996  24.224169  0.132491  0.085017  
    99997  25.613836  0.047444  0.043660  
    99998  25.274899  0.035128  0.034086  
    99999  25.699642  0.020565  0.020517  
    
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
          <td>29.803037</td>
          <td>2.465268</td>
          <td>27.100189</td>
          <td>0.230176</td>
          <td>26.182570</td>
          <td>0.092603</td>
          <td>25.149455</td>
          <td>0.060617</td>
          <td>24.763020</td>
          <td>0.082373</td>
          <td>23.957978</td>
          <td>0.091087</td>
          <td>0.179137</td>
          <td>0.150946</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.555466</td>
          <td>0.393568</td>
          <td>27.397908</td>
          <td>0.293616</td>
          <td>26.806528</td>
          <td>0.159161</td>
          <td>26.369421</td>
          <td>0.175765</td>
          <td>25.742372</td>
          <td>0.192230</td>
          <td>25.478970</td>
          <td>0.329151</td>
          <td>0.052724</td>
          <td>0.051298</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.105895</td>
          <td>0.140284</td>
          <td>25.105302</td>
          <td>0.111224</td>
          <td>24.357945</td>
          <td>0.129147</td>
          <td>0.058445</td>
          <td>0.053692</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.286635</td>
          <td>0.671497</td>
          <td>27.739215</td>
          <td>0.384609</td>
          <td>27.110240</td>
          <td>0.205835</td>
          <td>26.226088</td>
          <td>0.155545</td>
          <td>25.708954</td>
          <td>0.186886</td>
          <td>24.994537</td>
          <td>0.221905</td>
          <td>0.065644</td>
          <td>0.056715</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.997726</td>
          <td>0.252369</td>
          <td>26.089216</td>
          <td>0.096897</td>
          <td>25.921282</td>
          <td>0.073549</td>
          <td>25.772999</td>
          <td>0.105061</td>
          <td>25.262087</td>
          <td>0.127467</td>
          <td>25.111532</td>
          <td>0.244475</td>
          <td>0.043958</td>
          <td>0.022187</td>
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
          <td>26.181337</td>
          <td>0.292977</td>
          <td>26.562253</td>
          <td>0.146112</td>
          <td>25.479596</td>
          <td>0.049713</td>
          <td>25.138205</td>
          <td>0.060015</td>
          <td>24.885517</td>
          <td>0.091752</td>
          <td>24.803145</td>
          <td>0.189019</td>
          <td>0.004769</td>
          <td>0.003336</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.760760</td>
          <td>0.173114</td>
          <td>26.093441</td>
          <td>0.085619</td>
          <td>25.158843</td>
          <td>0.061124</td>
          <td>24.748848</td>
          <td>0.081349</td>
          <td>24.252175</td>
          <td>0.117819</td>
          <td>0.132491</td>
          <td>0.085017</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.435296</td>
          <td>0.358473</td>
          <td>26.751431</td>
          <td>0.171748</td>
          <td>26.228115</td>
          <td>0.096381</td>
          <td>26.119994</td>
          <td>0.141999</td>
          <td>26.607017</td>
          <td>0.387709</td>
          <td>25.083358</td>
          <td>0.238859</td>
          <td>0.047444</td>
          <td>0.043660</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.076655</td>
          <td>0.269171</td>
          <td>26.433900</td>
          <td>0.130811</td>
          <td>26.182024</td>
          <td>0.092558</td>
          <td>25.896471</td>
          <td>0.117010</td>
          <td>25.501962</td>
          <td>0.156721</td>
          <td>24.947469</td>
          <td>0.213368</td>
          <td>0.035128</td>
          <td>0.034086</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.744355</td>
          <td>0.454461</td>
          <td>26.725272</td>
          <td>0.167969</td>
          <td>26.461605</td>
          <td>0.118194</td>
          <td>26.042929</td>
          <td>0.132863</td>
          <td>25.733532</td>
          <td>0.190803</td>
          <td>25.280683</td>
          <td>0.280726</td>
          <td>0.020565</td>
          <td>0.020517</td>
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
          <td>27.253486</td>
          <td>0.755878</td>
          <td>26.995759</td>
          <td>0.259656</td>
          <td>26.145937</td>
          <td>0.114685</td>
          <td>25.161395</td>
          <td>0.079350</td>
          <td>24.690130</td>
          <td>0.098904</td>
          <td>23.851851</td>
          <td>0.106991</td>
          <td>0.179137</td>
          <td>0.150946</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.569768</td>
          <td>0.887801</td>
          <td>27.196598</td>
          <td>0.286743</td>
          <td>26.423366</td>
          <td>0.135314</td>
          <td>26.385386</td>
          <td>0.211260</td>
          <td>25.768616</td>
          <td>0.231083</td>
          <td>25.309271</td>
          <td>0.337295</td>
          <td>0.052724</td>
          <td>0.051298</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.827382</td>
          <td>0.536933</td>
          <td>29.552466</td>
          <td>1.404971</td>
          <td>27.625896</td>
          <td>0.366651</td>
          <td>26.146400</td>
          <td>0.172970</td>
          <td>24.972694</td>
          <td>0.117524</td>
          <td>24.446084</td>
          <td>0.165780</td>
          <td>0.058445</td>
          <td>0.053692</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.141314</td>
          <td>0.274985</td>
          <td>27.222605</td>
          <td>0.266125</td>
          <td>26.069525</td>
          <td>0.162339</td>
          <td>25.570053</td>
          <td>0.196435</td>
          <td>24.777841</td>
          <td>0.219705</td>
          <td>0.065644</td>
          <td>0.056715</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.834181</td>
          <td>0.537385</td>
          <td>26.025916</td>
          <td>0.106111</td>
          <td>25.940393</td>
          <td>0.088352</td>
          <td>25.827520</td>
          <td>0.130714</td>
          <td>25.567665</td>
          <td>0.194444</td>
          <td>24.976797</td>
          <td>0.256821</td>
          <td>0.043958</td>
          <td>0.022187</td>
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
          <td>28.173719</td>
          <td>1.259598</td>
          <td>26.166672</td>
          <td>0.119508</td>
          <td>25.379733</td>
          <td>0.053590</td>
          <td>25.008257</td>
          <td>0.063418</td>
          <td>24.731596</td>
          <td>0.094202</td>
          <td>24.877320</td>
          <td>0.235710</td>
          <td>0.004769</td>
          <td>0.003336</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.254849</td>
          <td>0.736665</td>
          <td>26.915093</td>
          <td>0.233821</td>
          <td>25.953319</td>
          <td>0.092652</td>
          <td>25.215708</td>
          <td>0.079445</td>
          <td>24.799865</td>
          <td>0.104094</td>
          <td>24.287255</td>
          <td>0.149064</td>
          <td>0.132491</td>
          <td>0.085017</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.977119</td>
          <td>0.596448</td>
          <td>27.659331</td>
          <td>0.412178</td>
          <td>26.357479</td>
          <td>0.127555</td>
          <td>27.174895</td>
          <td>0.398267</td>
          <td>26.120057</td>
          <td>0.307174</td>
          <td>25.233966</td>
          <td>0.317085</td>
          <td>0.047444</td>
          <td>0.043660</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.426541</td>
          <td>0.396041</td>
          <td>26.007184</td>
          <td>0.104386</td>
          <td>26.240390</td>
          <td>0.114883</td>
          <td>26.040885</td>
          <td>0.157050</td>
          <td>25.626136</td>
          <td>0.204225</td>
          <td>24.849072</td>
          <td>0.231159</td>
          <td>0.035128</td>
          <td>0.034086</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.412194</td>
          <td>0.390997</td>
          <td>26.873895</td>
          <td>0.218591</td>
          <td>26.712550</td>
          <td>0.172073</td>
          <td>26.314161</td>
          <td>0.197509</td>
          <td>25.575844</td>
          <td>0.195287</td>
          <td>25.752134</td>
          <td>0.471022</td>
          <td>0.020565</td>
          <td>0.020517</td>
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
          <td>27.040538</td>
          <td>0.665824</td>
          <td>26.487414</td>
          <td>0.174083</td>
          <td>26.062273</td>
          <td>0.109581</td>
          <td>25.034143</td>
          <td>0.072970</td>
          <td>24.662533</td>
          <td>0.099235</td>
          <td>24.074035</td>
          <td>0.133440</td>
          <td>0.179137</td>
          <td>0.150946</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.672834</td>
          <td>1.544586</td>
          <td>28.296311</td>
          <td>0.597111</td>
          <td>26.656066</td>
          <td>0.144817</td>
          <td>26.152825</td>
          <td>0.151470</td>
          <td>25.613994</td>
          <td>0.178435</td>
          <td>24.847166</td>
          <td>0.203183</td>
          <td>0.052724</td>
          <td>0.051298</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.481622</td>
          <td>1.261147</td>
          <td>28.304778</td>
          <td>0.546471</td>
          <td>26.155262</td>
          <td>0.152651</td>
          <td>25.134785</td>
          <td>0.118846</td>
          <td>24.225678</td>
          <td>0.120090</td>
          <td>0.058445</td>
          <td>0.053692</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.761418</td>
          <td>1.466415</td>
          <td>27.485018</td>
          <td>0.293218</td>
          <td>26.067937</td>
          <td>0.142716</td>
          <td>25.612425</td>
          <td>0.180464</td>
          <td>25.167379</td>
          <td>0.268232</td>
          <td>0.065644</td>
          <td>0.056715</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.189644</td>
          <td>0.297990</td>
          <td>26.260178</td>
          <td>0.114079</td>
          <td>25.966687</td>
          <td>0.077812</td>
          <td>25.882803</td>
          <td>0.117580</td>
          <td>25.226073</td>
          <td>0.125527</td>
          <td>24.908234</td>
          <td>0.209797</td>
          <td>0.043958</td>
          <td>0.022187</td>
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
          <td>28.935068</td>
          <td>1.726761</td>
          <td>26.526739</td>
          <td>0.141745</td>
          <td>25.508393</td>
          <td>0.051013</td>
          <td>24.999640</td>
          <td>0.053082</td>
          <td>24.810921</td>
          <td>0.085944</td>
          <td>24.728700</td>
          <td>0.177522</td>
          <td>0.004769</td>
          <td>0.003336</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.217667</td>
          <td>0.693804</td>
          <td>26.581909</td>
          <td>0.167843</td>
          <td>26.088325</td>
          <td>0.098203</td>
          <td>25.208512</td>
          <td>0.074131</td>
          <td>24.788949</td>
          <td>0.097092</td>
          <td>24.372035</td>
          <td>0.150891</td>
          <td>0.132491</td>
          <td>0.085017</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.409615</td>
          <td>0.740623</td>
          <td>26.650500</td>
          <td>0.161294</td>
          <td>26.438254</td>
          <td>0.119009</td>
          <td>26.095749</td>
          <td>0.143046</td>
          <td>26.027172</td>
          <td>0.250086</td>
          <td>25.301806</td>
          <td>0.293150</td>
          <td>0.047444</td>
          <td>0.043660</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.271473</td>
          <td>0.318129</td>
          <td>26.212745</td>
          <td>0.109451</td>
          <td>26.108907</td>
          <td>0.088198</td>
          <td>25.775276</td>
          <td>0.107047</td>
          <td>25.612207</td>
          <td>0.174858</td>
          <td>24.933206</td>
          <td>0.214201</td>
          <td>0.035128</td>
          <td>0.034086</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.125810</td>
          <td>0.281151</td>
          <td>26.731945</td>
          <td>0.169737</td>
          <td>26.592317</td>
          <td>0.133128</td>
          <td>26.237913</td>
          <td>0.158046</td>
          <td>25.626720</td>
          <td>0.175273</td>
          <td>25.475542</td>
          <td>0.330018</td>
          <td>0.020565</td>
          <td>0.020517</td>
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
