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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fb390ca7850>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.952162</td>
          <td>0.203460</td>
          <td>26.042255</td>
          <td>0.081842</td>
          <td>25.355850</td>
          <td>0.072780</td>
          <td>25.042958</td>
          <td>0.105330</td>
          <td>24.748956</td>
          <td>0.180554</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.350693</td>
          <td>0.605309</td>
          <td>27.380984</td>
          <td>0.257612</td>
          <td>26.931811</td>
          <td>0.280497</td>
          <td>26.614371</td>
          <td>0.389922</td>
          <td>25.537110</td>
          <td>0.344648</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.060296</td>
          <td>0.573005</td>
          <td>25.864225</td>
          <td>0.079513</td>
          <td>24.800130</td>
          <td>0.027281</td>
          <td>23.852903</td>
          <td>0.019440</td>
          <td>23.146700</td>
          <td>0.019895</td>
          <td>22.865907</td>
          <td>0.034639</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.000860</td>
          <td>0.421065</td>
          <td>26.484026</td>
          <td>0.193652</td>
          <td>26.023690</td>
          <td>0.243050</td>
          <td>26.442497</td>
          <td>0.673669</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.205672</td>
          <td>0.298768</td>
          <td>25.794040</td>
          <td>0.074740</td>
          <td>25.437291</td>
          <td>0.047881</td>
          <td>24.741138</td>
          <td>0.042187</td>
          <td>24.437844</td>
          <td>0.061782</td>
          <td>23.850901</td>
          <td>0.082893</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.339122</td>
          <td>0.120499</td>
          <td>26.071208</td>
          <td>0.083958</td>
          <td>26.114551</td>
          <td>0.141335</td>
          <td>26.154290</td>
          <td>0.270504</td>
          <td>27.133426</td>
          <td>1.047618</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.450852</td>
          <td>0.362862</td>
          <td>26.682602</td>
          <td>0.161971</td>
          <td>27.090062</td>
          <td>0.202382</td>
          <td>26.398999</td>
          <td>0.180230</td>
          <td>25.899893</td>
          <td>0.219353</td>
          <td>25.265980</td>
          <td>0.277397</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.339328</td>
          <td>0.280035</td>
          <td>27.075808</td>
          <td>0.199975</td>
          <td>26.025474</td>
          <td>0.130872</td>
          <td>25.903008</td>
          <td>0.219923</td>
          <td>25.336492</td>
          <td>0.293685</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.952537</td>
          <td>1.029154</td>
          <td>27.359376</td>
          <td>0.284620</td>
          <td>26.440673</td>
          <td>0.116060</td>
          <td>25.739676</td>
          <td>0.102042</td>
          <td>25.483292</td>
          <td>0.154236</td>
          <td>25.415493</td>
          <td>0.312918</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.114198</td>
          <td>0.277505</td>
          <td>26.543009</td>
          <td>0.143715</td>
          <td>26.195525</td>
          <td>0.093662</td>
          <td>25.762182</td>
          <td>0.104072</td>
          <td>25.194267</td>
          <td>0.120182</td>
          <td>25.083505</td>
          <td>0.238888</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>26.799442</td>
          <td>0.522625</td>
          <td>26.403717</td>
          <td>0.146650</td>
          <td>26.350350</td>
          <td>0.125896</td>
          <td>25.263766</td>
          <td>0.079495</td>
          <td>25.105507</td>
          <td>0.130504</td>
          <td>24.600679</td>
          <td>0.187040</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.661980</td>
          <td>0.833444</td>
          <td>28.076917</td>
          <td>0.511576</td>
          <td>26.922811</td>
          <td>0.324868</td>
          <td>26.512980</td>
          <td>0.415431</td>
          <td>26.008385</td>
          <td>0.567645</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.890541</td>
          <td>0.566127</td>
          <td>25.877445</td>
          <td>0.094736</td>
          <td>24.760289</td>
          <td>0.031684</td>
          <td>23.839569</td>
          <td>0.023187</td>
          <td>23.148935</td>
          <td>0.023876</td>
          <td>22.867510</td>
          <td>0.042033</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.632904</td>
          <td>0.855039</td>
          <td>28.082051</td>
          <td>0.542810</td>
          <td>26.803567</td>
          <td>0.314524</td>
          <td>25.901829</td>
          <td>0.272155</td>
          <td>26.204285</td>
          <td>0.687612</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.663789</td>
          <td>0.472939</td>
          <td>25.874013</td>
          <td>0.092593</td>
          <td>25.380771</td>
          <td>0.053655</td>
          <td>24.772807</td>
          <td>0.051482</td>
          <td>24.442112</td>
          <td>0.073018</td>
          <td>23.755926</td>
          <td>0.090177</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.606721</td>
          <td>0.459308</td>
          <td>26.336586</td>
          <td>0.141023</td>
          <td>26.136728</td>
          <td>0.106736</td>
          <td>26.168553</td>
          <td>0.178096</td>
          <td>25.925728</td>
          <td>0.265894</td>
          <td>24.985587</td>
          <td>0.262894</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.414523</td>
          <td>0.801660</td>
          <td>27.230299</td>
          <td>0.293448</td>
          <td>26.755558</td>
          <td>0.178948</td>
          <td>26.927041</td>
          <td>0.327174</td>
          <td>26.528489</td>
          <td>0.421831</td>
          <td>25.586298</td>
          <td>0.416553</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.954913</td>
          <td>0.516677</td>
          <td>26.739143</td>
          <td>0.177948</td>
          <td>26.869365</td>
          <td>0.315015</td>
          <td>25.980066</td>
          <td>0.275818</td>
          <td>26.032535</td>
          <td>0.583624</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.597159</td>
          <td>0.459154</td>
          <td>27.126588</td>
          <td>0.276022</td>
          <td>26.482910</td>
          <td>0.145589</td>
          <td>25.772886</td>
          <td>0.128200</td>
          <td>25.352056</td>
          <td>0.166310</td>
          <td>26.426983</td>
          <td>0.776299</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.542947</td>
          <td>0.434610</td>
          <td>26.574210</td>
          <td>0.171143</td>
          <td>26.117866</td>
          <td>0.103851</td>
          <td>25.764169</td>
          <td>0.124489</td>
          <td>25.060868</td>
          <td>0.126804</td>
          <td>25.182899</td>
          <td>0.305256</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.595125</td>
          <td>0.150309</td>
          <td>25.966670</td>
          <td>0.076570</td>
          <td>25.362475</td>
          <td>0.073218</td>
          <td>25.054965</td>
          <td>0.106456</td>
          <td>24.711255</td>
          <td>0.174896</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.070577</td>
          <td>0.972605</td>
          <td>27.472948</td>
          <td>0.277920</td>
          <td>27.561328</td>
          <td>0.459427</td>
          <td>26.326329</td>
          <td>0.311087</td>
          <td>25.083939</td>
          <td>0.239195</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.255982</td>
          <td>0.328011</td>
          <td>25.976720</td>
          <td>0.094346</td>
          <td>24.807836</td>
          <td>0.029814</td>
          <td>23.884220</td>
          <td>0.021704</td>
          <td>23.164738</td>
          <td>0.021885</td>
          <td>22.823440</td>
          <td>0.036346</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.993461</td>
          <td>1.063808</td>
          <td>27.568094</td>
          <td>0.367358</td>
          <td>26.596446</td>
          <td>0.265170</td>
          <td>26.993283</td>
          <td>0.623023</td>
          <td>24.854497</td>
          <td>0.245914</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.170148</td>
          <td>0.290609</td>
          <td>25.960112</td>
          <td>0.086623</td>
          <td>25.417629</td>
          <td>0.047120</td>
          <td>24.837823</td>
          <td>0.046036</td>
          <td>24.341901</td>
          <td>0.056822</td>
          <td>23.732725</td>
          <td>0.074793</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.074891</td>
          <td>0.283023</td>
          <td>26.387337</td>
          <td>0.134479</td>
          <td>26.252626</td>
          <td>0.106514</td>
          <td>26.025693</td>
          <td>0.141931</td>
          <td>25.675275</td>
          <td>0.195888</td>
          <td>25.594164</td>
          <td>0.387460</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.405570</td>
          <td>0.734327</td>
          <td>26.864609</td>
          <td>0.191635</td>
          <td>26.503846</td>
          <td>0.124625</td>
          <td>26.271926</td>
          <td>0.164502</td>
          <td>26.256561</td>
          <td>0.298343</td>
          <td>25.769352</td>
          <td>0.419034</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.342348</td>
          <td>0.716262</td>
          <td>27.404058</td>
          <td>0.306825</td>
          <td>26.970907</td>
          <td>0.191859</td>
          <td>26.773625</td>
          <td>0.258567</td>
          <td>25.834005</td>
          <td>0.217480</td>
          <td>25.005125</td>
          <td>0.234845</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.936640</td>
          <td>0.221050</td>
          <td>26.602147</td>
          <td>0.149401</td>
          <td>25.913262</td>
          <td>0.133633</td>
          <td>25.488074</td>
          <td>0.173089</td>
          <td>25.022984</td>
          <td>0.253994</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.542982</td>
          <td>0.399234</td>
          <td>26.554061</td>
          <td>0.149979</td>
          <td>26.080793</td>
          <td>0.088041</td>
          <td>25.567245</td>
          <td>0.091366</td>
          <td>25.305231</td>
          <td>0.137473</td>
          <td>25.030804</td>
          <td>0.237559</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
