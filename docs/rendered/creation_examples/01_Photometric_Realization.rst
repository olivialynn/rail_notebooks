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

    <pzflow.flow.Flow at 0x7f94413bc940>



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
    0      23.994413  0.007981  0.005163  
    1      25.391064  0.127483  0.073575  
    2      24.304707  0.041552  0.024101  
    3      25.291103  0.045158  0.023983  
    4      25.096743  0.044560  0.043657  
    ...          ...       ...       ...  
    99995  24.737946  0.045271  0.039906  
    99996  24.224169  0.038964  0.019828  
    99997  25.613836  0.219965  0.137666  
    99998  25.274899  0.146996  0.103390  
    99999  25.699642  0.115770  0.059833  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.578077</td>
          <td>0.148111</td>
          <td>25.975958</td>
          <td>0.077190</td>
          <td>25.213836</td>
          <td>0.064179</td>
          <td>24.655566</td>
          <td>0.074918</td>
          <td>24.094724</td>
          <td>0.102695</td>
          <td>0.007981</td>
          <td>0.005163</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.719786</td>
          <td>0.446132</td>
          <td>28.513498</td>
          <td>0.677862</td>
          <td>26.699704</td>
          <td>0.145229</td>
          <td>26.284090</td>
          <td>0.163452</td>
          <td>25.842230</td>
          <td>0.209045</td>
          <td>25.437656</td>
          <td>0.318506</td>
          <td>0.127483</td>
          <td>0.073575</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.397534</td>
          <td>0.564948</td>
          <td>26.046343</td>
          <td>0.133255</td>
          <td>24.986369</td>
          <td>0.100241</td>
          <td>24.233029</td>
          <td>0.115873</td>
          <td>0.041552</td>
          <td>0.024101</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.707356</td>
          <td>0.772169</td>
          <td>28.061337</td>
          <td>0.440867</td>
          <td>26.229230</td>
          <td>0.155964</td>
          <td>25.554785</td>
          <td>0.163957</td>
          <td>25.722191</td>
          <td>0.398155</td>
          <td>0.045158</td>
          <td>0.023983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.313090</td>
          <td>0.325536</td>
          <td>26.115251</td>
          <td>0.099131</td>
          <td>25.818154</td>
          <td>0.067133</td>
          <td>25.636476</td>
          <td>0.093212</td>
          <td>25.308310</td>
          <td>0.132670</td>
          <td>25.106368</td>
          <td>0.243437</td>
          <td>0.044560</td>
          <td>0.043657</td>
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
          <td>26.608225</td>
          <td>0.409857</td>
          <td>26.251964</td>
          <td>0.111703</td>
          <td>25.358817</td>
          <td>0.044659</td>
          <td>25.055688</td>
          <td>0.055777</td>
          <td>24.817162</td>
          <td>0.086397</td>
          <td>24.767314</td>
          <td>0.183382</td>
          <td>0.045271</td>
          <td>0.039906</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.823059</td>
          <td>0.481990</td>
          <td>26.731215</td>
          <td>0.168821</td>
          <td>25.962834</td>
          <td>0.076300</td>
          <td>25.086566</td>
          <td>0.057327</td>
          <td>24.934764</td>
          <td>0.095807</td>
          <td>24.203699</td>
          <td>0.112950</td>
          <td>0.038964</td>
          <td>0.019828</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.156861</td>
          <td>0.613609</td>
          <td>26.871566</td>
          <td>0.190131</td>
          <td>26.340077</td>
          <td>0.106309</td>
          <td>26.068477</td>
          <td>0.135828</td>
          <td>26.307173</td>
          <td>0.306086</td>
          <td>25.087102</td>
          <td>0.239599</td>
          <td>0.219965</td>
          <td>0.137666</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.139325</td>
          <td>0.283209</td>
          <td>26.021889</td>
          <td>0.091341</td>
          <td>26.032445</td>
          <td>0.081137</td>
          <td>25.924349</td>
          <td>0.119882</td>
          <td>25.526125</td>
          <td>0.159993</td>
          <td>25.138475</td>
          <td>0.249955</td>
          <td>0.146996</td>
          <td>0.103390</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.045043</td>
          <td>0.566782</td>
          <td>26.809633</td>
          <td>0.180439</td>
          <td>26.623488</td>
          <td>0.135997</td>
          <td>26.330453</td>
          <td>0.170039</td>
          <td>25.923032</td>
          <td>0.223618</td>
          <td>25.275563</td>
          <td>0.279563</td>
          <td>0.115770</td>
          <td>0.059833</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.317219</td>
          <td>0.136146</td>
          <td>26.037406</td>
          <td>0.095831</td>
          <td>25.239461</td>
          <td>0.077818</td>
          <td>24.726157</td>
          <td>0.093763</td>
          <td>23.882118</td>
          <td>0.100713</td>
          <td>0.007981</td>
          <td>0.005163</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.839319</td>
          <td>0.550122</td>
          <td>27.709773</td>
          <td>0.437914</td>
          <td>26.748094</td>
          <td>0.183224</td>
          <td>26.366361</td>
          <td>0.213336</td>
          <td>26.193278</td>
          <td>0.334063</td>
          <td>25.881011</td>
          <td>0.533275</td>
          <td>0.127483</td>
          <td>0.073575</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.133507</td>
          <td>0.664011</td>
          <td>29.411521</td>
          <td>1.299734</td>
          <td>28.952046</td>
          <td>0.928821</td>
          <td>25.840992</td>
          <td>0.132219</td>
          <td>24.740413</td>
          <td>0.095300</td>
          <td>24.196509</td>
          <td>0.132911</td>
          <td>0.041552</td>
          <td>0.024101</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.384822</td>
          <td>0.695826</td>
          <td>27.779381</td>
          <td>0.410650</td>
          <td>26.421805</td>
          <td>0.216776</td>
          <td>25.075030</td>
          <td>0.127658</td>
          <td>25.515300</td>
          <td>0.394525</td>
          <td>0.045158</td>
          <td>0.023983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.187135</td>
          <td>0.328999</td>
          <td>26.212680</td>
          <td>0.125098</td>
          <td>25.913780</td>
          <td>0.086522</td>
          <td>25.774162</td>
          <td>0.125127</td>
          <td>25.559232</td>
          <td>0.193527</td>
          <td>25.130504</td>
          <td>0.291699</td>
          <td>0.044560</td>
          <td>0.043657</td>
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
          <td>30.514678</td>
          <td>3.252878</td>
          <td>26.432320</td>
          <td>0.151107</td>
          <td>25.379210</td>
          <td>0.053897</td>
          <td>24.983992</td>
          <td>0.062467</td>
          <td>24.789700</td>
          <td>0.099732</td>
          <td>24.638082</td>
          <td>0.194205</td>
          <td>0.045271</td>
          <td>0.039906</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.217439</td>
          <td>0.336215</td>
          <td>26.998492</td>
          <td>0.242728</td>
          <td>26.039872</td>
          <td>0.096336</td>
          <td>25.273732</td>
          <td>0.080466</td>
          <td>24.837339</td>
          <td>0.103676</td>
          <td>24.516512</td>
          <td>0.174729</td>
          <td>0.038964</td>
          <td>0.019828</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.815791</td>
          <td>0.564358</td>
          <td>26.481468</td>
          <td>0.171423</td>
          <td>26.432390</td>
          <td>0.149241</td>
          <td>26.221341</td>
          <td>0.201567</td>
          <td>26.174809</td>
          <td>0.349282</td>
          <td>25.994474</td>
          <td>0.611040</td>
          <td>0.219965</td>
          <td>0.137666</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.132319</td>
          <td>0.325082</td>
          <td>26.214702</td>
          <td>0.130491</td>
          <td>26.016731</td>
          <td>0.099094</td>
          <td>25.630375</td>
          <td>0.115658</td>
          <td>25.545197</td>
          <td>0.199711</td>
          <td>25.574906</td>
          <td>0.430816</td>
          <td>0.146996</td>
          <td>0.103390</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.870926</td>
          <td>0.560153</td>
          <td>26.826331</td>
          <td>0.214914</td>
          <td>26.430956</td>
          <td>0.138730</td>
          <td>26.465810</td>
          <td>0.230061</td>
          <td>26.043222</td>
          <td>0.294321</td>
          <td>26.628836</td>
          <td>0.881714</td>
          <td>0.115770</td>
          <td>0.059833</td>
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
          <td>29.235759</td>
          <td>1.972765</td>
          <td>26.760982</td>
          <td>0.173236</td>
          <td>26.009523</td>
          <td>0.079561</td>
          <td>25.085892</td>
          <td>0.057330</td>
          <td>24.713641</td>
          <td>0.078910</td>
          <td>23.924467</td>
          <td>0.088498</td>
          <td>0.007981</td>
          <td>0.005163</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.304603</td>
          <td>1.329465</td>
          <td>27.465594</td>
          <td>0.342577</td>
          <td>26.772124</td>
          <td>0.174487</td>
          <td>26.646731</td>
          <td>0.250874</td>
          <td>25.788454</td>
          <td>0.225022</td>
          <td>25.498036</td>
          <td>0.375062</td>
          <td>0.127483</td>
          <td>0.073575</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.945593</td>
          <td>0.532165</td>
          <td>28.826731</td>
          <td>0.842713</td>
          <td>28.474596</td>
          <td>0.604288</td>
          <td>26.167916</td>
          <td>0.150326</td>
          <td>25.161434</td>
          <td>0.118589</td>
          <td>24.398857</td>
          <td>0.135917</td>
          <td>0.041552</td>
          <td>0.024101</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.981647</td>
          <td>0.187827</td>
          <td>26.218243</td>
          <td>0.157269</td>
          <td>25.597674</td>
          <td>0.172936</td>
          <td>26.232377</td>
          <td>0.590119</td>
          <td>0.045158</td>
          <td>0.023983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.781153</td>
          <td>0.474352</td>
          <td>26.313706</td>
          <td>0.120510</td>
          <td>25.857328</td>
          <td>0.071332</td>
          <td>25.724706</td>
          <td>0.103462</td>
          <td>25.219736</td>
          <td>0.126031</td>
          <td>26.093175</td>
          <td>0.537786</td>
          <td>0.044560</td>
          <td>0.043657</td>
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
          <td>28.230299</td>
          <td>1.220322</td>
          <td>26.408100</td>
          <td>0.130597</td>
          <td>25.467529</td>
          <td>0.050401</td>
          <td>25.170009</td>
          <td>0.063333</td>
          <td>24.847319</td>
          <td>0.090890</td>
          <td>24.672397</td>
          <td>0.173344</td>
          <td>0.045271</td>
          <td>0.039906</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.588138</td>
          <td>0.151021</td>
          <td>26.093790</td>
          <td>0.086749</td>
          <td>25.095567</td>
          <td>0.058576</td>
          <td>24.977198</td>
          <td>0.100712</td>
          <td>23.987031</td>
          <td>0.094689</td>
          <td>0.038964</td>
          <td>0.019828</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.367030</td>
          <td>0.418398</td>
          <td>26.502448</td>
          <td>0.182093</td>
          <td>26.387571</td>
          <td>0.150382</td>
          <td>26.067886</td>
          <td>0.185505</td>
          <td>25.607911</td>
          <td>0.230616</td>
          <td>25.029648</td>
          <td>0.307088</td>
          <td>0.219965</td>
          <td>0.137666</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.967862</td>
          <td>1.132113</td>
          <td>26.057474</td>
          <td>0.110308</td>
          <td>25.846969</td>
          <td>0.082385</td>
          <td>25.955199</td>
          <td>0.147742</td>
          <td>25.604007</td>
          <td>0.202842</td>
          <td>24.683089</td>
          <td>0.203711</td>
          <td>0.146996</td>
          <td>0.103390</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.802267</td>
          <td>0.504365</td>
          <td>27.040525</td>
          <td>0.238112</td>
          <td>26.397086</td>
          <td>0.123414</td>
          <td>26.094967</td>
          <td>0.153971</td>
          <td>25.787852</td>
          <td>0.219723</td>
          <td>25.307264</td>
          <td>0.315394</td>
          <td>0.115770</td>
          <td>0.059833</td>
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
