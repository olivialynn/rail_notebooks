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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fd7f293df90>



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
    0      23.994413  0.065268  0.042989  
    1      25.391064  0.138437  0.108518  
    2      24.304707  0.046053  0.036641  
    3      25.291103  0.114180  0.110290  
    4      25.096743  0.023507  0.016115  
    ...          ...       ...       ...  
    99995  24.737946  0.057077  0.028696  
    99996  24.224169  0.060234  0.042150  
    99997  25.613836  0.099414  0.072659  
    99998  25.274899  0.035306  0.030024  
    99999  25.699642  0.268103  0.250848  
    
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
          <td>26.892317</td>
          <td>0.507299</td>
          <td>26.534671</td>
          <td>0.142688</td>
          <td>25.917903</td>
          <td>0.073329</td>
          <td>25.237380</td>
          <td>0.065532</td>
          <td>24.675844</td>
          <td>0.076272</td>
          <td>24.084376</td>
          <td>0.101769</td>
          <td>0.065268</td>
          <td>0.042989</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.344858</td>
          <td>0.698719</td>
          <td>27.697210</td>
          <td>0.372259</td>
          <td>26.558135</td>
          <td>0.128524</td>
          <td>26.242000</td>
          <td>0.157678</td>
          <td>25.574826</td>
          <td>0.166783</td>
          <td>25.031235</td>
          <td>0.228774</td>
          <td>0.138437</td>
          <td>0.108518</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.283352</td>
          <td>1.243330</td>
          <td>27.877940</td>
          <td>0.427841</td>
          <td>27.345515</td>
          <td>0.250224</td>
          <td>25.881618</td>
          <td>0.115507</td>
          <td>24.912837</td>
          <td>0.093980</td>
          <td>24.356007</td>
          <td>0.128931</td>
          <td>0.046053</td>
          <td>0.036641</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.470077</td>
          <td>0.311134</td>
          <td>27.368620</td>
          <td>0.255015</td>
          <td>26.262278</td>
          <td>0.160436</td>
          <td>25.499639</td>
          <td>0.156410</td>
          <td>25.269163</td>
          <td>0.278115</td>
          <td>0.114180</td>
          <td>0.110290</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.447373</td>
          <td>0.361877</td>
          <td>26.077181</td>
          <td>0.095880</td>
          <td>25.843784</td>
          <td>0.068674</td>
          <td>25.683187</td>
          <td>0.097113</td>
          <td>25.553118</td>
          <td>0.163724</td>
          <td>25.046377</td>
          <td>0.231664</td>
          <td>0.023507</td>
          <td>0.016115</td>
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
          <td>26.327970</td>
          <td>0.119338</td>
          <td>25.444297</td>
          <td>0.048179</td>
          <td>25.016190</td>
          <td>0.053855</td>
          <td>24.807002</td>
          <td>0.085628</td>
          <td>24.996186</td>
          <td>0.222210</td>
          <td>0.057077</td>
          <td>0.028696</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.585069</td>
          <td>1.457761</td>
          <td>26.968530</td>
          <td>0.206269</td>
          <td>26.066513</td>
          <td>0.083611</td>
          <td>25.120945</td>
          <td>0.059103</td>
          <td>24.938693</td>
          <td>0.096138</td>
          <td>24.282672</td>
          <td>0.120985</td>
          <td>0.060234</td>
          <td>0.042150</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.763388</td>
          <td>0.917318</td>
          <td>26.935601</td>
          <td>0.200653</td>
          <td>26.468956</td>
          <td>0.118952</td>
          <td>26.235979</td>
          <td>0.156868</td>
          <td>25.695043</td>
          <td>0.184702</td>
          <td>25.483952</td>
          <td>0.330455</td>
          <td>0.099414</td>
          <td>0.072659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.108241</td>
          <td>0.592903</td>
          <td>26.365620</td>
          <td>0.123302</td>
          <td>26.243094</td>
          <td>0.097655</td>
          <td>25.802295</td>
          <td>0.107786</td>
          <td>25.695120</td>
          <td>0.184714</td>
          <td>24.951045</td>
          <td>0.214006</td>
          <td>0.035306</td>
          <td>0.030024</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.357542</td>
          <td>0.337204</td>
          <td>26.865515</td>
          <td>0.189164</td>
          <td>26.721026</td>
          <td>0.147915</td>
          <td>26.283561</td>
          <td>0.163378</td>
          <td>25.991865</td>
          <td>0.236748</td>
          <td>25.867100</td>
          <td>0.444708</td>
          <td>0.268103</td>
          <td>0.250848</td>
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
          <td>26.648768</td>
          <td>0.182343</td>
          <td>26.094156</td>
          <td>0.101741</td>
          <td>25.236203</td>
          <td>0.078418</td>
          <td>24.556322</td>
          <td>0.081574</td>
          <td>24.072846</td>
          <td>0.120179</td>
          <td>0.065268</td>
          <td>0.042989</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.259522</td>
          <td>1.350019</td>
          <td>27.076616</td>
          <td>0.269176</td>
          <td>26.643343</td>
          <td>0.170030</td>
          <td>26.196041</td>
          <td>0.187595</td>
          <td>25.713195</td>
          <td>0.229351</td>
          <td>25.614607</td>
          <td>0.443223</td>
          <td>0.138437</td>
          <td>0.108518</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.748640</td>
          <td>2.540891</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.878736</td>
          <td>0.443459</td>
          <td>25.870765</td>
          <td>0.135934</td>
          <td>25.146508</td>
          <td>0.135989</td>
          <td>24.618994</td>
          <td>0.191050</td>
          <td>0.046053</td>
          <td>0.036641</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.297617</td>
          <td>0.672732</td>
          <td>27.998137</td>
          <td>0.499567</td>
          <td>26.059919</td>
          <td>0.165682</td>
          <td>25.275840</td>
          <td>0.157306</td>
          <td>24.863370</td>
          <td>0.242428</td>
          <td>0.114180</td>
          <td>0.110290</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.150308</td>
          <td>0.318369</td>
          <td>26.098039</td>
          <td>0.112719</td>
          <td>25.919971</td>
          <td>0.086541</td>
          <td>25.713472</td>
          <td>0.118072</td>
          <td>25.708252</td>
          <td>0.218175</td>
          <td>25.514403</td>
          <td>0.393169</td>
          <td>0.023507</td>
          <td>0.016115</td>
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
          <td>26.938230</td>
          <td>0.580147</td>
          <td>26.221975</td>
          <td>0.126143</td>
          <td>25.355935</td>
          <td>0.052834</td>
          <td>25.052149</td>
          <td>0.066405</td>
          <td>25.015558</td>
          <td>0.121535</td>
          <td>24.571636</td>
          <td>0.183747</td>
          <td>0.057077</td>
          <td>0.028696</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.380400</td>
          <td>0.786179</td>
          <td>26.760655</td>
          <td>0.200165</td>
          <td>26.039233</td>
          <td>0.096848</td>
          <td>25.217627</td>
          <td>0.077048</td>
          <td>24.785062</td>
          <td>0.099620</td>
          <td>24.095448</td>
          <td>0.122415</td>
          <td>0.060234</td>
          <td>0.042150</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.113934</td>
          <td>0.663496</td>
          <td>26.897230</td>
          <td>0.227429</td>
          <td>26.525954</td>
          <td>0.150139</td>
          <td>26.493311</td>
          <td>0.234737</td>
          <td>26.148048</td>
          <td>0.319327</td>
          <td>26.168433</td>
          <td>0.648621</td>
          <td>0.099414</td>
          <td>0.072659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.057416</td>
          <td>0.296023</td>
          <td>26.232761</td>
          <td>0.126959</td>
          <td>26.182956</td>
          <td>0.109227</td>
          <td>25.839650</td>
          <td>0.132030</td>
          <td>25.706726</td>
          <td>0.218366</td>
          <td>25.280546</td>
          <td>0.328029</td>
          <td>0.035306</td>
          <td>0.030024</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.556928</td>
          <td>0.495191</td>
          <td>27.153430</td>
          <td>0.321628</td>
          <td>26.699920</td>
          <td>0.203581</td>
          <td>25.876001</td>
          <td>0.163904</td>
          <td>25.775382</td>
          <td>0.274460</td>
          <td>25.237059</td>
          <td>0.375071</td>
          <td>0.268103</td>
          <td>0.250848</td>
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
          <td>30.802331</td>
          <td>3.425088</td>
          <td>26.626643</td>
          <td>0.159684</td>
          <td>25.996660</td>
          <td>0.081806</td>
          <td>25.200442</td>
          <td>0.066136</td>
          <td>24.641434</td>
          <td>0.076993</td>
          <td>23.910647</td>
          <td>0.091044</td>
          <td>0.065268</td>
          <td>0.042989</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.483036</td>
          <td>0.360892</td>
          <td>26.583717</td>
          <td>0.155675</td>
          <td>26.715220</td>
          <td>0.277875</td>
          <td>25.524068</td>
          <td>0.188759</td>
          <td>25.407797</td>
          <td>0.365242</td>
          <td>0.138437</td>
          <td>0.108518</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.027360</td>
          <td>0.566994</td>
          <td>28.886890</td>
          <td>0.879753</td>
          <td>28.766092</td>
          <td>0.742601</td>
          <td>25.906148</td>
          <td>0.120847</td>
          <td>25.130732</td>
          <td>0.116333</td>
          <td>24.519499</td>
          <td>0.151953</td>
          <td>0.046053</td>
          <td>0.036641</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.945658</td>
          <td>0.575382</td>
          <td>28.515375</td>
          <td>0.749502</td>
          <td>27.002886</td>
          <td>0.216276</td>
          <td>26.163959</td>
          <td>0.171096</td>
          <td>25.370134</td>
          <td>0.161429</td>
          <td>25.010733</td>
          <td>0.259231</td>
          <td>0.114180</td>
          <td>0.110290</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.403276</td>
          <td>0.350772</td>
          <td>26.191018</td>
          <td>0.106423</td>
          <td>25.849594</td>
          <td>0.069412</td>
          <td>25.552533</td>
          <td>0.087080</td>
          <td>25.385982</td>
          <td>0.142631</td>
          <td>25.339602</td>
          <td>0.295965</td>
          <td>0.023507</td>
          <td>0.016115</td>
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
          <td>26.612078</td>
          <td>0.417809</td>
          <td>26.527929</td>
          <td>0.145133</td>
          <td>25.426710</td>
          <td>0.048734</td>
          <td>25.060959</td>
          <td>0.057651</td>
          <td>24.838295</td>
          <td>0.090405</td>
          <td>24.964087</td>
          <td>0.222108</td>
          <td>0.057077</td>
          <td>0.028696</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.206665</td>
          <td>0.647895</td>
          <td>26.766605</td>
          <td>0.179212</td>
          <td>26.152128</td>
          <td>0.093396</td>
          <td>25.145679</td>
          <td>0.062716</td>
          <td>24.754604</td>
          <td>0.084707</td>
          <td>24.295591</td>
          <td>0.126853</td>
          <td>0.060234</td>
          <td>0.042150</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.305967</td>
          <td>0.343564</td>
          <td>26.801088</td>
          <td>0.193709</td>
          <td>26.327531</td>
          <td>0.115289</td>
          <td>26.190683</td>
          <td>0.165811</td>
          <td>26.539141</td>
          <td>0.399059</td>
          <td>25.550759</td>
          <td>0.379410</td>
          <td>0.099414</td>
          <td>0.072659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.457405</td>
          <td>0.367977</td>
          <td>26.262612</td>
          <td>0.114149</td>
          <td>25.933603</td>
          <td>0.075439</td>
          <td>25.943845</td>
          <td>0.123763</td>
          <td>25.947473</td>
          <td>0.231315</td>
          <td>25.826677</td>
          <td>0.436892</td>
          <td>0.035306</td>
          <td>0.030024</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.416860</td>
          <td>0.501122</td>
          <td>26.449274</td>
          <td>0.208516</td>
          <td>26.342552</td>
          <td>0.175970</td>
          <td>26.447252</td>
          <td>0.307647</td>
          <td>26.202103</td>
          <td>0.443581</td>
          <td>27.038488</td>
          <td>1.383097</td>
          <td>0.268103</td>
          <td>0.250848</td>
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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_24_0.png


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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_25_0.png


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
