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

    <pzflow.flow.Flow at 0x7f3ec8607310>



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
    0      23.994413  0.003347  0.002825  
    1      25.391064  0.022434  0.020820  
    2      24.304707  0.093261  0.083942  
    3      25.291103  0.097157  0.087489  
    4      25.096743  0.019431  0.011510  
    ...          ...       ...       ...  
    99995  24.737946  0.068017  0.042547  
    99996  24.224169  0.114148  0.075619  
    99997  25.613836  0.045812  0.045291  
    99998  25.274899  0.147482  0.114450  
    99999  25.699642  0.147770  0.084618  
    
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
          <td>26.827642</td>
          <td>0.483633</td>
          <td>26.940973</td>
          <td>0.201560</td>
          <td>26.009349</td>
          <td>0.079500</td>
          <td>25.171229</td>
          <td>0.061799</td>
          <td>24.774079</td>
          <td>0.083180</td>
          <td>23.978570</td>
          <td>0.092750</td>
          <td>0.003347</td>
          <td>0.002825</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.816368</td>
          <td>0.181470</td>
          <td>27.022113</td>
          <td>0.191139</td>
          <td>26.374406</td>
          <td>0.176510</td>
          <td>25.745937</td>
          <td>0.192809</td>
          <td>25.690660</td>
          <td>0.388577</td>
          <td>0.022434</td>
          <td>0.020820</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.785961</td>
          <td>1.448971</td>
          <td>28.023880</td>
          <td>0.428515</td>
          <td>25.933033</td>
          <td>0.120790</td>
          <td>25.259158</td>
          <td>0.127144</td>
          <td>24.218775</td>
          <td>0.114443</td>
          <td>0.093261</td>
          <td>0.083942</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.898079</td>
          <td>2.550867</td>
          <td>29.161039</td>
          <td>1.026457</td>
          <td>27.212806</td>
          <td>0.224230</td>
          <td>26.342361</td>
          <td>0.171770</td>
          <td>25.875080</td>
          <td>0.214862</td>
          <td>25.068856</td>
          <td>0.236014</td>
          <td>0.097157</td>
          <td>0.087489</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.654193</td>
          <td>0.424500</td>
          <td>26.105296</td>
          <td>0.098271</td>
          <td>25.967631</td>
          <td>0.076624</td>
          <td>25.732010</td>
          <td>0.101359</td>
          <td>25.331021</td>
          <td>0.135299</td>
          <td>25.086880</td>
          <td>0.239555</td>
          <td>0.019431</td>
          <td>0.011510</td>
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
          <td>27.205809</td>
          <td>0.634993</td>
          <td>26.470277</td>
          <td>0.134987</td>
          <td>25.423378</td>
          <td>0.047293</td>
          <td>25.055798</td>
          <td>0.055783</td>
          <td>24.897531</td>
          <td>0.092725</td>
          <td>24.704971</td>
          <td>0.173942</td>
          <td>0.068017</td>
          <td>0.042547</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.750074</td>
          <td>0.456418</td>
          <td>26.874912</td>
          <td>0.190668</td>
          <td>26.133271</td>
          <td>0.088674</td>
          <td>25.183333</td>
          <td>0.062466</td>
          <td>24.805215</td>
          <td>0.085493</td>
          <td>24.265092</td>
          <td>0.119150</td>
          <td>0.114148</td>
          <td>0.075619</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.583691</td>
          <td>0.402214</td>
          <td>26.450246</td>
          <td>0.132672</td>
          <td>26.254341</td>
          <td>0.098623</td>
          <td>26.559668</td>
          <td>0.206356</td>
          <td>26.483137</td>
          <td>0.351992</td>
          <td>25.974006</td>
          <td>0.481806</td>
          <td>0.045812</td>
          <td>0.045291</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.238224</td>
          <td>0.306670</td>
          <td>26.083008</td>
          <td>0.096371</td>
          <td>25.994441</td>
          <td>0.078461</td>
          <td>25.744492</td>
          <td>0.102473</td>
          <td>25.581461</td>
          <td>0.167728</td>
          <td>25.033529</td>
          <td>0.229210</td>
          <td>0.147482</td>
          <td>0.114450</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.345985</td>
          <td>0.334137</td>
          <td>26.614782</td>
          <td>0.152846</td>
          <td>26.796653</td>
          <td>0.157823</td>
          <td>26.418307</td>
          <td>0.183200</td>
          <td>25.628324</td>
          <td>0.174549</td>
          <td>25.365982</td>
          <td>0.300741</td>
          <td>0.147770</td>
          <td>0.084618</td>
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
          <td>25.848676</td>
          <td>0.249213</td>
          <td>26.735891</td>
          <td>0.194512</td>
          <td>26.192851</td>
          <td>0.109779</td>
          <td>25.253261</td>
          <td>0.078762</td>
          <td>24.738679</td>
          <td>0.094787</td>
          <td>23.878011</td>
          <td>0.100339</td>
          <td>0.003347</td>
          <td>0.002825</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.877021</td>
          <td>1.065803</td>
          <td>28.347775</td>
          <td>0.677112</td>
          <td>26.452854</td>
          <td>0.137777</td>
          <td>26.016131</td>
          <td>0.153373</td>
          <td>25.731589</td>
          <td>0.222499</td>
          <td>25.445013</td>
          <td>0.372639</td>
          <td>0.022434</td>
          <td>0.020820</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.522489</td>
          <td>0.376529</td>
          <td>28.001951</td>
          <td>0.494692</td>
          <td>25.893691</td>
          <td>0.141527</td>
          <td>24.942671</td>
          <td>0.116263</td>
          <td>24.521438</td>
          <td>0.179483</td>
          <td>0.093261</td>
          <td>0.083942</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.711973</td>
          <td>0.398131</td>
          <td>26.327235</td>
          <td>0.205054</td>
          <td>25.591514</td>
          <td>0.203014</td>
          <td>25.120073</td>
          <td>0.295226</td>
          <td>0.097157</td>
          <td>0.087489</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.116983</td>
          <td>0.309905</td>
          <td>26.260013</td>
          <td>0.129667</td>
          <td>26.003285</td>
          <td>0.093070</td>
          <td>25.586764</td>
          <td>0.105667</td>
          <td>25.364252</td>
          <td>0.163137</td>
          <td>25.540534</td>
          <td>0.400984</td>
          <td>0.019431</td>
          <td>0.011510</td>
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
          <td>26.508679</td>
          <td>0.423656</td>
          <td>26.490593</td>
          <td>0.159483</td>
          <td>25.318836</td>
          <td>0.051328</td>
          <td>25.104207</td>
          <td>0.069823</td>
          <td>24.960579</td>
          <td>0.116316</td>
          <td>24.679661</td>
          <td>0.202034</td>
          <td>0.068017</td>
          <td>0.042547</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.578171</td>
          <td>0.452467</td>
          <td>26.696078</td>
          <td>0.193185</td>
          <td>26.122873</td>
          <td>0.106482</td>
          <td>25.177041</td>
          <td>0.076032</td>
          <td>24.722429</td>
          <td>0.096362</td>
          <td>23.972124</td>
          <td>0.112416</td>
          <td>0.114148</td>
          <td>0.075619</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.185816</td>
          <td>0.689497</td>
          <td>27.048623</td>
          <td>0.253746</td>
          <td>26.205229</td>
          <td>0.111744</td>
          <td>26.196994</td>
          <td>0.179909</td>
          <td>25.685581</td>
          <td>0.215238</td>
          <td>25.324400</td>
          <td>0.340686</td>
          <td>0.045812</td>
          <td>0.045291</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.331728</td>
          <td>0.381207</td>
          <td>26.011322</td>
          <td>0.109771</td>
          <td>26.117863</td>
          <td>0.108689</td>
          <td>25.514459</td>
          <td>0.104965</td>
          <td>25.987000</td>
          <td>0.288576</td>
          <td>25.150063</td>
          <td>0.310324</td>
          <td>0.147482</td>
          <td>0.114450</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.834379</td>
          <td>0.552110</td>
          <td>27.025100</td>
          <td>0.257355</td>
          <td>26.464599</td>
          <td>0.145442</td>
          <td>25.989476</td>
          <td>0.156848</td>
          <td>25.922718</td>
          <td>0.271575</td>
          <td>26.564623</td>
          <td>0.858309</td>
          <td>0.147770</td>
          <td>0.084618</td>
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
          <td>26.763367</td>
          <td>0.173517</td>
          <td>26.052938</td>
          <td>0.082627</td>
          <td>25.302354</td>
          <td>0.069424</td>
          <td>24.692917</td>
          <td>0.077441</td>
          <td>23.913640</td>
          <td>0.087615</td>
          <td>0.003347</td>
          <td>0.002825</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.470240</td>
          <td>0.762498</td>
          <td>27.276520</td>
          <td>0.267448</td>
          <td>26.626285</td>
          <td>0.137175</td>
          <td>26.288048</td>
          <td>0.165066</td>
          <td>26.571358</td>
          <td>0.379242</td>
          <td>24.923092</td>
          <td>0.210377</td>
          <td>0.022434</td>
          <td>0.020820</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.744672</td>
          <td>0.842199</td>
          <td>28.064990</td>
          <td>0.480013</td>
          <td>26.114130</td>
          <td>0.155924</td>
          <td>24.911383</td>
          <td>0.103342</td>
          <td>24.271796</td>
          <td>0.132271</td>
          <td>0.093261</td>
          <td>0.083942</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>31.935849</td>
          <td>4.580178</td>
          <td>27.723815</td>
          <td>0.411733</td>
          <td>27.581548</td>
          <td>0.333102</td>
          <td>26.458482</td>
          <td>0.210256</td>
          <td>25.250905</td>
          <td>0.139816</td>
          <td>25.165745</td>
          <td>0.282427</td>
          <td>0.097157</td>
          <td>0.087489</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.915020</td>
          <td>0.516847</td>
          <td>26.208884</td>
          <td>0.107906</td>
          <td>26.044361</td>
          <td>0.082279</td>
          <td>25.738740</td>
          <td>0.102328</td>
          <td>25.495361</td>
          <td>0.156363</td>
          <td>24.875524</td>
          <td>0.201584</td>
          <td>0.019431</td>
          <td>0.011510</td>
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
          <td>26.854540</td>
          <td>0.505598</td>
          <td>26.493796</td>
          <td>0.142739</td>
          <td>25.500697</td>
          <td>0.052831</td>
          <td>24.994777</td>
          <td>0.055223</td>
          <td>24.859041</td>
          <td>0.093444</td>
          <td>24.883469</td>
          <td>0.210719</td>
          <td>0.068017</td>
          <td>0.042547</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.738794</td>
          <td>0.484431</td>
          <td>26.637444</td>
          <td>0.171412</td>
          <td>25.980247</td>
          <td>0.086657</td>
          <td>25.176047</td>
          <td>0.069800</td>
          <td>24.963763</td>
          <td>0.109793</td>
          <td>24.333982</td>
          <td>0.141675</td>
          <td>0.114148</td>
          <td>0.075619</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.912578</td>
          <td>1.726024</td>
          <td>27.052710</td>
          <td>0.226326</td>
          <td>26.367950</td>
          <td>0.111937</td>
          <td>26.389874</td>
          <td>0.183865</td>
          <td>26.105113</td>
          <td>0.266557</td>
          <td>26.565083</td>
          <td>0.748040</td>
          <td>0.045812</td>
          <td>0.045291</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.372393</td>
          <td>0.386411</td>
          <td>26.306668</td>
          <td>0.138462</td>
          <td>26.126876</td>
          <td>0.106689</td>
          <td>25.950959</td>
          <td>0.149134</td>
          <td>25.546042</td>
          <td>0.195586</td>
          <td>24.800193</td>
          <td>0.227449</td>
          <td>0.147482</td>
          <td>0.114450</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.738251</td>
          <td>0.499509</td>
          <td>26.793694</td>
          <td>0.203957</td>
          <td>26.341269</td>
          <td>0.124685</td>
          <td>26.063089</td>
          <td>0.159089</td>
          <td>25.877573</td>
          <td>0.250243</td>
          <td>25.445798</td>
          <td>0.371671</td>
          <td>0.147770</td>
          <td>0.084618</td>
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
