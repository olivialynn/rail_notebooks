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

    <pzflow.flow.Flow at 0x7f4898c42890>



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
    0      23.994413  0.057158  0.034596  
    1      25.391064  0.017394  0.010414  
    2      24.304707  0.050330  0.046895  
    3      25.291103  0.068307  0.035949  
    4      25.096743  0.078518  0.071101  
    ...          ...       ...       ...  
    99995  24.737946  0.168877  0.113460  
    99996  24.224169  0.136178  0.099628  
    99997  25.613836  0.029812  0.015031  
    99998  25.274899  0.019218  0.009865  
    99999  25.699642  0.174269  0.160003  
    
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
          <td>28.032351</td>
          <td>1.078707</td>
          <td>26.918097</td>
          <td>0.197725</td>
          <td>25.898773</td>
          <td>0.072099</td>
          <td>25.168598</td>
          <td>0.061655</td>
          <td>24.730046</td>
          <td>0.080011</td>
          <td>23.834197</td>
          <td>0.081681</td>
          <td>0.057158</td>
          <td>0.034596</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.602445</td>
          <td>2.287204</td>
          <td>27.985397</td>
          <td>0.463993</td>
          <td>26.514255</td>
          <td>0.123726</td>
          <td>26.201467</td>
          <td>0.152298</td>
          <td>25.918196</td>
          <td>0.222720</td>
          <td>25.100288</td>
          <td>0.242220</td>
          <td>0.017394</td>
          <td>0.010414</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.420684</td>
          <td>1.192423</td>
          <td>28.201106</td>
          <td>0.489517</td>
          <td>26.016108</td>
          <td>0.129815</td>
          <td>25.117080</td>
          <td>0.112372</td>
          <td>24.231263</td>
          <td>0.115695</td>
          <td>0.050330</td>
          <td>0.046895</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.116321</td>
          <td>1.701601</td>
          <td>27.240114</td>
          <td>0.229371</td>
          <td>26.138711</td>
          <td>0.144305</td>
          <td>25.703050</td>
          <td>0.185956</td>
          <td>25.612966</td>
          <td>0.365796</td>
          <td>0.068307</td>
          <td>0.035949</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.887946</td>
          <td>0.230556</td>
          <td>26.267971</td>
          <td>0.113271</td>
          <td>25.900086</td>
          <td>0.072183</td>
          <td>25.580671</td>
          <td>0.088749</td>
          <td>25.543134</td>
          <td>0.162335</td>
          <td>24.950227</td>
          <td>0.213860</td>
          <td>0.078518</td>
          <td>0.071101</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.474296</td>
          <td>0.135456</td>
          <td>25.322777</td>
          <td>0.043253</td>
          <td>25.129339</td>
          <td>0.059545</td>
          <td>25.055290</td>
          <td>0.106472</td>
          <td>24.959615</td>
          <td>0.215542</td>
          <td>0.168877</td>
          <td>0.113460</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.559794</td>
          <td>1.439148</td>
          <td>26.834919</td>
          <td>0.184340</td>
          <td>25.979654</td>
          <td>0.077443</td>
          <td>25.175786</td>
          <td>0.062050</td>
          <td>24.960005</td>
          <td>0.097952</td>
          <td>24.315656</td>
          <td>0.124499</td>
          <td>0.136178</td>
          <td>0.099628</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.897215</td>
          <td>0.509127</td>
          <td>26.704414</td>
          <td>0.165012</td>
          <td>26.339952</td>
          <td>0.106298</td>
          <td>26.555096</td>
          <td>0.205567</td>
          <td>25.766147</td>
          <td>0.196118</td>
          <td>26.045241</td>
          <td>0.507865</td>
          <td>0.029812</td>
          <td>0.015031</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.562083</td>
          <td>0.395581</td>
          <td>26.303392</td>
          <td>0.116816</td>
          <td>26.239259</td>
          <td>0.097327</td>
          <td>25.735475</td>
          <td>0.101667</td>
          <td>25.224136</td>
          <td>0.123340</td>
          <td>26.108095</td>
          <td>0.531765</td>
          <td>0.019218</td>
          <td>0.009865</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.243960</td>
          <td>0.308082</td>
          <td>26.966339</td>
          <td>0.205891</td>
          <td>26.708689</td>
          <td>0.146355</td>
          <td>25.953209</td>
          <td>0.122926</td>
          <td>25.833801</td>
          <td>0.207575</td>
          <td>25.293681</td>
          <td>0.283699</td>
          <td>0.174269</td>
          <td>0.160003</td>
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
          <td>26.204923</td>
          <td>0.333885</td>
          <td>27.021329</td>
          <td>0.248232</td>
          <td>26.145966</td>
          <td>0.106166</td>
          <td>25.204215</td>
          <td>0.076016</td>
          <td>24.672650</td>
          <td>0.090121</td>
          <td>23.991822</td>
          <td>0.111683</td>
          <td>0.057158</td>
          <td>0.034596</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.947824</td>
          <td>0.509328</td>
          <td>26.743743</td>
          <td>0.176566</td>
          <td>26.063097</td>
          <td>0.159520</td>
          <td>26.268960</td>
          <td>0.343814</td>
          <td>24.918546</td>
          <td>0.244020</td>
          <td>0.017394</td>
          <td>0.010414</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.716586</td>
          <td>0.494263</td>
          <td>28.244537</td>
          <td>0.633345</td>
          <td>28.914381</td>
          <td>0.909963</td>
          <td>26.122923</td>
          <td>0.169106</td>
          <td>25.123144</td>
          <td>0.133552</td>
          <td>24.539822</td>
          <td>0.179061</td>
          <td>0.050330</td>
          <td>0.046895</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.602772</td>
          <td>0.906784</td>
          <td>27.807280</td>
          <td>0.462146</td>
          <td>27.209168</td>
          <td>0.262576</td>
          <td>26.300189</td>
          <td>0.196868</td>
          <td>25.261276</td>
          <td>0.150715</td>
          <td>25.048920</td>
          <td>0.273933</td>
          <td>0.068307</td>
          <td>0.035949</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.835946</td>
          <td>0.249938</td>
          <td>26.108587</td>
          <td>0.115542</td>
          <td>25.970052</td>
          <td>0.092025</td>
          <td>25.502576</td>
          <td>0.099991</td>
          <td>25.238223</td>
          <td>0.149029</td>
          <td>24.934978</td>
          <td>0.251669</td>
          <td>0.078518</td>
          <td>0.071101</td>
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
          <td>26.998344</td>
          <td>0.627649</td>
          <td>26.479196</td>
          <td>0.165714</td>
          <td>25.468499</td>
          <td>0.061933</td>
          <td>25.124482</td>
          <td>0.075229</td>
          <td>24.947881</td>
          <td>0.121403</td>
          <td>25.225801</td>
          <td>0.332517</td>
          <td>0.168877</td>
          <td>0.113460</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.817660</td>
          <td>1.053024</td>
          <td>27.010337</td>
          <td>0.254155</td>
          <td>26.031131</td>
          <td>0.099775</td>
          <td>25.204445</td>
          <td>0.079138</td>
          <td>24.749829</td>
          <td>0.100216</td>
          <td>24.629436</td>
          <td>0.200496</td>
          <td>0.136178</td>
          <td>0.099628</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.203014</td>
          <td>0.695510</td>
          <td>26.795846</td>
          <td>0.204880</td>
          <td>26.309368</td>
          <td>0.121723</td>
          <td>26.498493</td>
          <td>0.230472</td>
          <td>25.804107</td>
          <td>0.236354</td>
          <td>24.926223</td>
          <td>0.245850</td>
          <td>0.029812</td>
          <td>0.015031</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.686473</td>
          <td>0.481128</td>
          <td>26.202168</td>
          <td>0.123326</td>
          <td>25.916645</td>
          <td>0.086238</td>
          <td>25.711655</td>
          <td>0.117815</td>
          <td>25.407257</td>
          <td>0.169214</td>
          <td>25.410144</td>
          <td>0.362366</td>
          <td>0.019218</td>
          <td>0.009865</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.037791</td>
          <td>0.653815</td>
          <td>26.579553</td>
          <td>0.183925</td>
          <td>26.465880</td>
          <td>0.151474</td>
          <td>26.355393</td>
          <td>0.222351</td>
          <td>25.603570</td>
          <td>0.216835</td>
          <td>27.426557</td>
          <td>1.444370</td>
          <td>0.174269</td>
          <td>0.160003</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>27.305014</td>
          <td>0.278824</td>
          <td>26.125281</td>
          <td>0.090671</td>
          <td>25.124815</td>
          <td>0.061175</td>
          <td>24.692966</td>
          <td>0.079744</td>
          <td>24.005722</td>
          <td>0.097913</td>
          <td>0.057158</td>
          <td>0.034596</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.867449</td>
          <td>2.525177</td>
          <td>27.306960</td>
          <td>0.273380</td>
          <td>26.666187</td>
          <td>0.141484</td>
          <td>26.187668</td>
          <td>0.150936</td>
          <td>26.110946</td>
          <td>0.261779</td>
          <td>25.630171</td>
          <td>0.371694</td>
          <td>0.017394</td>
          <td>0.010414</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.483700</td>
          <td>0.779370</td>
          <td>28.120676</td>
          <td>0.524639</td>
          <td>28.292132</td>
          <td>0.537099</td>
          <td>26.141063</td>
          <td>0.149290</td>
          <td>24.996575</td>
          <td>0.104325</td>
          <td>24.284468</td>
          <td>0.125107</td>
          <td>0.050330</td>
          <td>0.046895</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.617113</td>
          <td>1.354211</td>
          <td>27.417434</td>
          <td>0.275022</td>
          <td>25.962400</td>
          <td>0.128931</td>
          <td>25.618202</td>
          <td>0.179586</td>
          <td>24.943607</td>
          <td>0.220875</td>
          <td>0.068307</td>
          <td>0.035949</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.622470</td>
          <td>0.432712</td>
          <td>25.840471</td>
          <td>0.082863</td>
          <td>26.036621</td>
          <td>0.087429</td>
          <td>25.759450</td>
          <td>0.111757</td>
          <td>25.357040</td>
          <td>0.148289</td>
          <td>24.880939</td>
          <td>0.216396</td>
          <td>0.078518</td>
          <td>0.071101</td>
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
          <td>28.320005</td>
          <td>1.397287</td>
          <td>26.398039</td>
          <td>0.153267</td>
          <td>25.354875</td>
          <td>0.055419</td>
          <td>25.125208</td>
          <td>0.074467</td>
          <td>24.808001</td>
          <td>0.106389</td>
          <td>24.891689</td>
          <td>0.251430</td>
          <td>0.168877</td>
          <td>0.113460</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.715767</td>
          <td>0.491307</td>
          <td>26.889007</td>
          <td>0.220957</td>
          <td>25.947996</td>
          <td>0.088502</td>
          <td>25.167351</td>
          <td>0.072933</td>
          <td>24.683215</td>
          <td>0.090200</td>
          <td>24.183844</td>
          <td>0.130822</td>
          <td>0.136178</td>
          <td>0.099628</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.694967</td>
          <td>0.439809</td>
          <td>26.501946</td>
          <td>0.139613</td>
          <td>26.365757</td>
          <td>0.109535</td>
          <td>26.276135</td>
          <td>0.163592</td>
          <td>25.654074</td>
          <td>0.179698</td>
          <td>26.425991</td>
          <td>0.670141</td>
          <td>0.029812</td>
          <td>0.015031</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.124406</td>
          <td>0.280375</td>
          <td>26.271898</td>
          <td>0.113968</td>
          <td>26.073530</td>
          <td>0.084397</td>
          <td>26.062743</td>
          <td>0.135599</td>
          <td>25.583989</td>
          <td>0.168604</td>
          <td>25.684642</td>
          <td>0.387891</td>
          <td>0.019218</td>
          <td>0.009865</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.873185</td>
          <td>1.129091</td>
          <td>26.998780</td>
          <td>0.267674</td>
          <td>26.730801</td>
          <td>0.195496</td>
          <td>26.373551</td>
          <td>0.232630</td>
          <td>26.034923</td>
          <td>0.317406</td>
          <td>25.862959</td>
          <td>0.564295</td>
          <td>0.174269</td>
          <td>0.160003</td>
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
