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

    <pzflow.flow.Flow at 0x7fcf72bfe2c0>



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
    0      23.994413  0.024419  0.019607  
    1      25.391064  0.103783  0.067770  
    2      24.304707  0.071790  0.065838  
    3      25.291103  0.007779  0.004755  
    4      25.096743  0.028729  0.024321  
    ...          ...       ...       ...  
    99995  24.737946  0.103658  0.070503  
    99996  24.224169  0.115430  0.080275  
    99997  25.613836  0.067637  0.047655  
    99998  25.274899  0.010536  0.010093  
    99999  25.699642  0.078324  0.041777  
    
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
          <td>27.534991</td>
          <td>0.793043</td>
          <td>26.584503</td>
          <td>0.148930</td>
          <td>26.141699</td>
          <td>0.089334</td>
          <td>25.157352</td>
          <td>0.061043</td>
          <td>24.678190</td>
          <td>0.076431</td>
          <td>24.149940</td>
          <td>0.107774</td>
          <td>0.024419</td>
          <td>0.019607</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.938303</td>
          <td>0.895790</td>
          <td>26.665108</td>
          <td>0.140967</td>
          <td>26.303558</td>
          <td>0.166188</td>
          <td>25.718299</td>
          <td>0.188367</td>
          <td>25.854576</td>
          <td>0.440517</td>
          <td>0.103783</td>
          <td>0.067770</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.374982</td>
          <td>0.615751</td>
          <td>27.966995</td>
          <td>0.410299</td>
          <td>25.834541</td>
          <td>0.110863</td>
          <td>24.981875</td>
          <td>0.099847</td>
          <td>24.284087</td>
          <td>0.121134</td>
          <td>0.071790</td>
          <td>0.065838</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.101943</td>
          <td>0.590260</td>
          <td>29.004540</td>
          <td>0.933488</td>
          <td>27.598573</td>
          <td>0.307297</td>
          <td>26.035441</td>
          <td>0.132005</td>
          <td>25.222839</td>
          <td>0.123202</td>
          <td>24.750303</td>
          <td>0.180761</td>
          <td>0.007779</td>
          <td>0.004755</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.984331</td>
          <td>0.249612</td>
          <td>26.052032</td>
          <td>0.093789</td>
          <td>25.912772</td>
          <td>0.072997</td>
          <td>25.747234</td>
          <td>0.102719</td>
          <td>25.449639</td>
          <td>0.149848</td>
          <td>25.288419</td>
          <td>0.282492</td>
          <td>0.028729</td>
          <td>0.024321</td>
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
          <td>26.046527</td>
          <td>0.262644</td>
          <td>26.204858</td>
          <td>0.107208</td>
          <td>25.430938</td>
          <td>0.047611</td>
          <td>24.941173</td>
          <td>0.050384</td>
          <td>24.764737</td>
          <td>0.082497</td>
          <td>24.814332</td>
          <td>0.190812</td>
          <td>0.103658</td>
          <td>0.070503</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.683024</td>
          <td>0.433900</td>
          <td>26.692698</td>
          <td>0.163372</td>
          <td>26.060749</td>
          <td>0.083188</td>
          <td>25.293501</td>
          <td>0.068873</td>
          <td>24.974770</td>
          <td>0.099228</td>
          <td>24.257652</td>
          <td>0.118382</td>
          <td>0.115430</td>
          <td>0.080275</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.827272</td>
          <td>0.483500</td>
          <td>26.938437</td>
          <td>0.201131</td>
          <td>26.272364</td>
          <td>0.100193</td>
          <td>26.142791</td>
          <td>0.144813</td>
          <td>25.940746</td>
          <td>0.226933</td>
          <td>25.776589</td>
          <td>0.415137</td>
          <td>0.067637</td>
          <td>0.047655</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.405177</td>
          <td>0.350103</td>
          <td>26.128054</td>
          <td>0.100249</td>
          <td>26.009349</td>
          <td>0.079500</td>
          <td>25.873177</td>
          <td>0.114661</td>
          <td>25.655264</td>
          <td>0.178585</td>
          <td>24.798966</td>
          <td>0.188354</td>
          <td>0.010536</td>
          <td>0.010093</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.211221</td>
          <td>0.637391</td>
          <td>26.848839</td>
          <td>0.186520</td>
          <td>26.468517</td>
          <td>0.118906</td>
          <td>26.250014</td>
          <td>0.158762</td>
          <td>26.138348</td>
          <td>0.267012</td>
          <td>25.603332</td>
          <td>0.363050</td>
          <td>0.078324</td>
          <td>0.041777</td>
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
          <td>26.690366</td>
          <td>0.482792</td>
          <td>26.616181</td>
          <td>0.176056</td>
          <td>25.989720</td>
          <td>0.092042</td>
          <td>25.111494</td>
          <td>0.069606</td>
          <td>24.594518</td>
          <td>0.083640</td>
          <td>23.922518</td>
          <td>0.104498</td>
          <td>0.024419</td>
          <td>0.019607</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.268614</td>
          <td>0.307970</td>
          <td>26.509619</td>
          <td>0.148089</td>
          <td>26.426992</td>
          <td>0.222233</td>
          <td>26.005403</td>
          <td>0.284825</td>
          <td>25.793673</td>
          <td>0.495989</td>
          <td>0.103783</td>
          <td>0.067770</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.615855</td>
          <td>0.917115</td>
          <td>30.328213</td>
          <td>2.021513</td>
          <td>27.858821</td>
          <td>0.440618</td>
          <td>26.052488</td>
          <td>0.160518</td>
          <td>24.924011</td>
          <td>0.113239</td>
          <td>24.115426</td>
          <td>0.125416</td>
          <td>0.071790</td>
          <td>0.065838</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.256475</td>
          <td>0.635060</td>
          <td>26.923380</td>
          <td>0.205334</td>
          <td>26.364972</td>
          <td>0.205850</td>
          <td>25.377955</td>
          <td>0.164939</td>
          <td>25.499801</td>
          <td>0.388320</td>
          <td>0.007779</td>
          <td>0.004755</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.420660</td>
          <td>0.393815</td>
          <td>26.251002</td>
          <td>0.128837</td>
          <td>25.945919</td>
          <td>0.088631</td>
          <td>25.441795</td>
          <td>0.093212</td>
          <td>25.837298</td>
          <td>0.243036</td>
          <td>24.574493</td>
          <td>0.183377</td>
          <td>0.028729</td>
          <td>0.024321</td>
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
          <td>27.226509</td>
          <td>0.716743</td>
          <td>26.381269</td>
          <td>0.147189</td>
          <td>25.483623</td>
          <td>0.060325</td>
          <td>24.958080</td>
          <td>0.062327</td>
          <td>24.882812</td>
          <td>0.110342</td>
          <td>24.636510</td>
          <td>0.197762</td>
          <td>0.103658</td>
          <td>0.070503</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.592083</td>
          <td>1.584770</td>
          <td>26.663949</td>
          <td>0.188294</td>
          <td>26.126727</td>
          <td>0.107016</td>
          <td>25.399060</td>
          <td>0.092613</td>
          <td>24.928194</td>
          <td>0.115530</td>
          <td>24.487254</td>
          <td>0.175451</td>
          <td>0.115430</td>
          <td>0.080275</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.829474</td>
          <td>0.538054</td>
          <td>26.776176</td>
          <td>0.203207</td>
          <td>26.266561</td>
          <td>0.118400</td>
          <td>26.565023</td>
          <td>0.245781</td>
          <td>26.006905</td>
          <td>0.281552</td>
          <td>25.927093</td>
          <td>0.540436</td>
          <td>0.067637</td>
          <td>0.047655</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.468428</td>
          <td>0.407991</td>
          <td>26.560030</td>
          <td>0.167664</td>
          <td>26.126281</td>
          <td>0.103611</td>
          <td>26.061514</td>
          <td>0.159250</td>
          <td>26.162817</td>
          <td>0.315941</td>
          <td>25.267255</td>
          <td>0.323592</td>
          <td>0.010536</td>
          <td>0.010093</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>30.102607</td>
          <td>2.871257</td>
          <td>26.718332</td>
          <td>0.193853</td>
          <td>26.341896</td>
          <td>0.126611</td>
          <td>26.273642</td>
          <td>0.193125</td>
          <td>26.893883</td>
          <td>0.557397</td>
          <td>25.883129</td>
          <td>0.524165</td>
          <td>0.078324</td>
          <td>0.041777</td>
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
          <td>28.024747</td>
          <td>1.077206</td>
          <td>26.844099</td>
          <td>0.186803</td>
          <td>25.992733</td>
          <td>0.078866</td>
          <td>25.169669</td>
          <td>0.062149</td>
          <td>24.695023</td>
          <td>0.078093</td>
          <td>23.975923</td>
          <td>0.093174</td>
          <td>0.024419</td>
          <td>0.019607</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.218648</td>
          <td>0.675093</td>
          <td>28.658290</td>
          <td>0.794807</td>
          <td>26.389992</td>
          <td>0.121793</td>
          <td>26.356328</td>
          <td>0.190920</td>
          <td>25.730538</td>
          <td>0.208042</td>
          <td>25.583877</td>
          <td>0.389472</td>
          <td>0.103783</td>
          <td>0.067770</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.736845</td>
          <td>1.457052</td>
          <td>27.474944</td>
          <td>0.294044</td>
          <td>25.820506</td>
          <td>0.116619</td>
          <td>25.006134</td>
          <td>0.108315</td>
          <td>24.153548</td>
          <td>0.115063</td>
          <td>0.071790</td>
          <td>0.065838</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.501362</td>
          <td>2.910181</td>
          <td>27.380980</td>
          <td>0.257747</td>
          <td>26.308881</td>
          <td>0.167040</td>
          <td>25.528274</td>
          <td>0.160375</td>
          <td>25.083325</td>
          <td>0.238984</td>
          <td>0.007779</td>
          <td>0.004755</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.241979</td>
          <td>0.309453</td>
          <td>26.261911</td>
          <td>0.113604</td>
          <td>25.917077</td>
          <td>0.073983</td>
          <td>25.666143</td>
          <td>0.096634</td>
          <td>25.515153</td>
          <td>0.159975</td>
          <td>25.015934</td>
          <td>0.228011</td>
          <td>0.028729</td>
          <td>0.024321</td>
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
          <td>26.630542</td>
          <td>0.168065</td>
          <td>25.496034</td>
          <td>0.055559</td>
          <td>25.009020</td>
          <td>0.059193</td>
          <td>24.704422</td>
          <td>0.086077</td>
          <td>24.758089</td>
          <td>0.200033</td>
          <td>0.103658</td>
          <td>0.070503</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.207820</td>
          <td>0.679334</td>
          <td>26.725655</td>
          <td>0.185610</td>
          <td>25.963763</td>
          <td>0.085895</td>
          <td>25.228788</td>
          <td>0.073569</td>
          <td>24.677031</td>
          <td>0.085873</td>
          <td>24.178450</td>
          <td>0.124569</td>
          <td>0.115430</td>
          <td>0.080275</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.463048</td>
          <td>0.376557</td>
          <td>27.149366</td>
          <td>0.248549</td>
          <td>26.390313</td>
          <td>0.116056</td>
          <td>26.024665</td>
          <td>0.136876</td>
          <td>26.077448</td>
          <td>0.264694</td>
          <td>25.756919</td>
          <td>0.425557</td>
          <td>0.067637</td>
          <td>0.047655</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.485730</td>
          <td>0.373198</td>
          <td>26.329052</td>
          <td>0.119598</td>
          <td>25.970260</td>
          <td>0.076914</td>
          <td>26.060468</td>
          <td>0.135093</td>
          <td>26.018160</td>
          <td>0.242272</td>
          <td>25.321819</td>
          <td>0.290629</td>
          <td>0.010536</td>
          <td>0.010093</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.818314</td>
          <td>0.189575</td>
          <td>26.574529</td>
          <td>0.136937</td>
          <td>26.624559</td>
          <td>0.228917</td>
          <td>25.534336</td>
          <td>0.169125</td>
          <td>26.218747</td>
          <td>0.600604</td>
          <td>0.078324</td>
          <td>0.041777</td>
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
