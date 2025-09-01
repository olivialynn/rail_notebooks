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

    <pzflow.flow.Flow at 0x7f28ae21a4a0>



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
    0      23.994413  0.060305  0.045466  
    1      25.391064  0.181170  0.154375  
    2      24.304707  0.025690  0.025557  
    3      25.291103  0.038815  0.020711  
    4      25.096743  0.099939  0.082653  
    ...          ...       ...       ...  
    99995  24.737946  0.005628  0.005016  
    99996  24.224169  0.158467  0.098020  
    99997  25.613836  0.049865  0.032095  
    99998  25.274899  0.143035  0.090499  
    99999  25.699642  0.049644  0.049129  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>30.313269</td>
          <td>2.932603</td>
          <td>26.944925</td>
          <td>0.202229</td>
          <td>25.906787</td>
          <td>0.072612</td>
          <td>25.300886</td>
          <td>0.069325</td>
          <td>24.650142</td>
          <td>0.074559</td>
          <td>23.934504</td>
          <td>0.089226</td>
          <td>0.060305</td>
          <td>0.045466</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.753491</td>
          <td>0.457590</td>
          <td>27.131584</td>
          <td>0.236234</td>
          <td>26.346614</td>
          <td>0.106918</td>
          <td>26.353214</td>
          <td>0.173362</td>
          <td>25.909988</td>
          <td>0.221205</td>
          <td>26.008471</td>
          <td>0.494278</td>
          <td>0.181170</td>
          <td>0.154375</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.839640</td>
          <td>1.488769</td>
          <td>28.373386</td>
          <td>0.555220</td>
          <td>25.785863</td>
          <td>0.106249</td>
          <td>25.072180</td>
          <td>0.108055</td>
          <td>24.420890</td>
          <td>0.136370</td>
          <td>0.025690</td>
          <td>0.025557</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.107347</td>
          <td>0.592527</td>
          <td>29.847494</td>
          <td>1.494634</td>
          <td>27.265525</td>
          <td>0.234251</td>
          <td>26.143918</td>
          <td>0.144953</td>
          <td>25.884665</td>
          <td>0.216587</td>
          <td>25.367935</td>
          <td>0.301213</td>
          <td>0.038815</td>
          <td>0.020711</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.576212</td>
          <td>0.399908</td>
          <td>26.166115</td>
          <td>0.103641</td>
          <td>26.063289</td>
          <td>0.083374</td>
          <td>25.797829</td>
          <td>0.107366</td>
          <td>25.445070</td>
          <td>0.149262</td>
          <td>25.188120</td>
          <td>0.260340</td>
          <td>0.099939</td>
          <td>0.082653</td>
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
          <td>26.517483</td>
          <td>0.140593</td>
          <td>25.427332</td>
          <td>0.047459</td>
          <td>25.099051</td>
          <td>0.057966</td>
          <td>24.791401</td>
          <td>0.084459</td>
          <td>24.508108</td>
          <td>0.147009</td>
          <td>0.005628</td>
          <td>0.005016</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.049397</td>
          <td>0.568553</td>
          <td>26.489287</td>
          <td>0.137218</td>
          <td>25.922577</td>
          <td>0.073633</td>
          <td>25.223500</td>
          <td>0.064731</td>
          <td>25.055940</td>
          <td>0.106533</td>
          <td>24.278518</td>
          <td>0.120549</td>
          <td>0.158467</td>
          <td>0.098020</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.140076</td>
          <td>0.606400</td>
          <td>26.645445</td>
          <td>0.156911</td>
          <td>26.178838</td>
          <td>0.092299</td>
          <td>26.222888</td>
          <td>0.155119</td>
          <td>26.026876</td>
          <td>0.243689</td>
          <td>25.847032</td>
          <td>0.438008</td>
          <td>0.049865</td>
          <td>0.032095</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.802611</td>
          <td>0.214790</td>
          <td>26.414892</td>
          <td>0.128678</td>
          <td>26.086459</td>
          <td>0.085094</td>
          <td>25.875619</td>
          <td>0.114905</td>
          <td>25.805276</td>
          <td>0.202672</td>
          <td>26.050430</td>
          <td>0.509806</td>
          <td>0.143035</td>
          <td>0.090499</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.912127</td>
          <td>0.196736</td>
          <td>26.434195</td>
          <td>0.115407</td>
          <td>26.342641</td>
          <td>0.171811</td>
          <td>25.438588</td>
          <td>0.148433</td>
          <td>27.014237</td>
          <td>0.975407</td>
          <td>0.049644</td>
          <td>0.049129</td>
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
          <td>27.435450</td>
          <td>0.815101</td>
          <td>26.623328</td>
          <td>0.178356</td>
          <td>26.036090</td>
          <td>0.096630</td>
          <td>25.133430</td>
          <td>0.071561</td>
          <td>24.734294</td>
          <td>0.095332</td>
          <td>24.211387</td>
          <td>0.135408</td>
          <td>0.060305</td>
          <td>0.045466</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.124336</td>
          <td>0.694156</td>
          <td>27.054616</td>
          <td>0.273011</td>
          <td>26.542159</td>
          <td>0.161823</td>
          <td>26.760646</td>
          <td>0.309837</td>
          <td>26.620205</td>
          <td>0.485474</td>
          <td>25.168252</td>
          <td>0.324358</td>
          <td>0.181170</td>
          <td>0.154375</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.268974</td>
          <td>0.641565</td>
          <td>27.226439</td>
          <td>0.264387</td>
          <td>25.893355</td>
          <td>0.138099</td>
          <td>24.991174</td>
          <td>0.118441</td>
          <td>24.331321</td>
          <td>0.149024</td>
          <td>0.025690</td>
          <td>0.025557</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.738756</td>
          <td>0.436604</td>
          <td>27.148810</td>
          <td>0.248336</td>
          <td>26.280909</td>
          <td>0.192410</td>
          <td>25.444369</td>
          <td>0.175058</td>
          <td>25.175328</td>
          <td>0.301480</td>
          <td>0.038815</td>
          <td>0.020711</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.828491</td>
          <td>0.543240</td>
          <td>26.260260</td>
          <td>0.132843</td>
          <td>25.895876</td>
          <td>0.087002</td>
          <td>25.685173</td>
          <td>0.118364</td>
          <td>25.306952</td>
          <td>0.159467</td>
          <td>25.398276</td>
          <td>0.367988</td>
          <td>0.099939</td>
          <td>0.082653</td>
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
          <td>26.589373</td>
          <td>0.171860</td>
          <td>25.383910</td>
          <td>0.053791</td>
          <td>25.064101</td>
          <td>0.066637</td>
          <td>25.042131</td>
          <td>0.123539</td>
          <td>24.665043</td>
          <td>0.197477</td>
          <td>0.005628</td>
          <td>0.005016</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.134200</td>
          <td>0.685161</td>
          <td>26.851565</td>
          <td>0.224712</td>
          <td>26.057978</td>
          <td>0.103087</td>
          <td>25.271226</td>
          <td>0.084738</td>
          <td>24.861805</td>
          <td>0.111530</td>
          <td>24.181108</td>
          <td>0.138120</td>
          <td>0.158467</td>
          <td>0.098020</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.562041</td>
          <td>2.373803</td>
          <td>26.723859</td>
          <td>0.193540</td>
          <td>26.203157</td>
          <td>0.111424</td>
          <td>26.601000</td>
          <td>0.251800</td>
          <td>25.851549</td>
          <td>0.246722</td>
          <td>25.104797</td>
          <td>0.285531</td>
          <td>0.049865</td>
          <td>0.032095</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.501105</td>
          <td>0.431265</td>
          <td>26.356697</td>
          <td>0.146721</td>
          <td>25.849999</td>
          <td>0.085108</td>
          <td>25.865849</td>
          <td>0.141002</td>
          <td>25.908694</td>
          <td>0.268397</td>
          <td>25.467364</td>
          <td>0.394746</td>
          <td>0.143035</td>
          <td>0.090499</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.967193</td>
          <td>0.276109</td>
          <td>26.681323</td>
          <td>0.187096</td>
          <td>26.534262</td>
          <td>0.148742</td>
          <td>26.425346</td>
          <td>0.218232</td>
          <td>25.972418</td>
          <td>0.272952</td>
          <td>25.809716</td>
          <td>0.494532</td>
          <td>0.049644</td>
          <td>0.049129</td>
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

.. parsed-literal::

    




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
          <td>26.642625</td>
          <td>0.430374</td>
          <td>26.998785</td>
          <td>0.218162</td>
          <td>26.161110</td>
          <td>0.094318</td>
          <td>25.246976</td>
          <td>0.068745</td>
          <td>24.660964</td>
          <td>0.078144</td>
          <td>23.979214</td>
          <td>0.096454</td>
          <td>0.060305</td>
          <td>0.045466</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.166485</td>
          <td>0.650258</td>
          <td>26.719239</td>
          <td>0.193922</td>
          <td>26.295234</td>
          <td>0.218340</td>
          <td>25.545123</td>
          <td>0.213034</td>
          <td>25.346679</td>
          <td>0.384184</td>
          <td>0.181170</td>
          <td>0.154375</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.884894</td>
          <td>2.356399</td>
          <td>29.721020</td>
          <td>1.306336</td>
          <td>26.050744</td>
          <td>0.134991</td>
          <td>25.171837</td>
          <td>0.118896</td>
          <td>24.221291</td>
          <td>0.115744</td>
          <td>0.025690</td>
          <td>0.025557</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.746568</td>
          <td>0.913563</td>
          <td>28.022576</td>
          <td>0.481614</td>
          <td>27.274665</td>
          <td>0.238914</td>
          <td>26.167538</td>
          <td>0.149901</td>
          <td>25.817943</td>
          <td>0.207373</td>
          <td>25.151729</td>
          <td>0.255877</td>
          <td>0.038815</td>
          <td>0.020711</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.550918</td>
          <td>0.418026</td>
          <td>26.076994</td>
          <td>0.104849</td>
          <td>26.061259</td>
          <td>0.092206</td>
          <td>25.611970</td>
          <td>0.101502</td>
          <td>25.562268</td>
          <td>0.182126</td>
          <td>25.306846</td>
          <td>0.315858</td>
          <td>0.099939</td>
          <td>0.082653</td>
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
          <td>26.903935</td>
          <td>0.511759</td>
          <td>26.519430</td>
          <td>0.140875</td>
          <td>25.463475</td>
          <td>0.049026</td>
          <td>25.027186</td>
          <td>0.054406</td>
          <td>24.900165</td>
          <td>0.092976</td>
          <td>24.959979</td>
          <td>0.215690</td>
          <td>0.005628</td>
          <td>0.005016</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.730379</td>
          <td>0.989236</td>
          <td>26.746341</td>
          <td>0.200595</td>
          <td>25.951562</td>
          <td>0.091111</td>
          <td>25.119773</td>
          <td>0.071833</td>
          <td>24.859590</td>
          <td>0.108025</td>
          <td>24.370319</td>
          <td>0.157621</td>
          <td>0.158467</td>
          <td>0.098020</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.268279</td>
          <td>0.318785</td>
          <td>26.590911</td>
          <td>0.152735</td>
          <td>26.165047</td>
          <td>0.093333</td>
          <td>26.242261</td>
          <td>0.161497</td>
          <td>25.962261</td>
          <td>0.236122</td>
          <td>25.887726</td>
          <td>0.461091</td>
          <td>0.049865</td>
          <td>0.032095</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.418730</td>
          <td>0.392340</td>
          <td>26.287878</td>
          <td>0.132547</td>
          <td>26.160997</td>
          <td>0.106565</td>
          <td>25.821830</td>
          <td>0.129262</td>
          <td>25.351471</td>
          <td>0.160937</td>
          <td>25.049522</td>
          <td>0.270956</td>
          <td>0.143035</td>
          <td>0.090499</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.689485</td>
          <td>0.444519</td>
          <td>26.648918</td>
          <td>0.161708</td>
          <td>26.636018</td>
          <td>0.141870</td>
          <td>26.619840</td>
          <td>0.224020</td>
          <td>25.748146</td>
          <td>0.199182</td>
          <td>27.233299</td>
          <td>1.134887</td>
          <td>0.049644</td>
          <td>0.049129</td>
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
