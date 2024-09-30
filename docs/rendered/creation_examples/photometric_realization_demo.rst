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

    <pzflow.flow.Flow at 0x7f8df4597a00>



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
          <td>27.154724</td>
          <td>0.612687</td>
          <td>26.975968</td>
          <td>0.207557</td>
          <td>26.116461</td>
          <td>0.087372</td>
          <td>25.392793</td>
          <td>0.075197</td>
          <td>25.083932</td>
          <td>0.109169</td>
          <td>25.148869</td>
          <td>0.252098</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.317476</td>
          <td>0.685820</td>
          <td>28.356881</td>
          <td>0.607957</td>
          <td>27.281207</td>
          <td>0.237308</td>
          <td>27.118391</td>
          <td>0.325845</td>
          <td>26.450495</td>
          <td>0.343057</td>
          <td>26.105302</td>
          <td>0.530685</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.642942</td>
          <td>0.420877</td>
          <td>25.906014</td>
          <td>0.082494</td>
          <td>24.785824</td>
          <td>0.026942</td>
          <td>23.839170</td>
          <td>0.019216</td>
          <td>23.129535</td>
          <td>0.019609</td>
          <td>22.808751</td>
          <td>0.032936</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.681926</td>
          <td>0.871628</td>
          <td>28.334440</td>
          <td>0.598396</td>
          <td>27.262484</td>
          <td>0.233662</td>
          <td>26.313803</td>
          <td>0.167646</td>
          <td>26.427813</td>
          <td>0.336963</td>
          <td>25.645048</td>
          <td>0.375063</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.799341</td>
          <td>0.214206</td>
          <td>25.685266</td>
          <td>0.067893</td>
          <td>25.463910</td>
          <td>0.049026</td>
          <td>24.760717</td>
          <td>0.042926</td>
          <td>24.371614</td>
          <td>0.058257</td>
          <td>23.711133</td>
          <td>0.073270</td>
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
          <td>26.277941</td>
          <td>0.316558</td>
          <td>26.276159</td>
          <td>0.114081</td>
          <td>26.025669</td>
          <td>0.080653</td>
          <td>26.248439</td>
          <td>0.158549</td>
          <td>26.223482</td>
          <td>0.286132</td>
          <td>26.095820</td>
          <td>0.527030</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.028073</td>
          <td>0.559919</td>
          <td>27.104084</td>
          <td>0.230920</td>
          <td>27.232861</td>
          <td>0.227995</td>
          <td>26.312250</td>
          <td>0.167424</td>
          <td>26.490395</td>
          <td>0.354005</td>
          <td>25.077848</td>
          <td>0.237774</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.339902</td>
          <td>0.280166</td>
          <td>26.612244</td>
          <td>0.134682</td>
          <td>26.653464</td>
          <td>0.223159</td>
          <td>26.066925</td>
          <td>0.251851</td>
          <td>25.371788</td>
          <td>0.302147</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.882242</td>
          <td>0.503553</td>
          <td>27.240521</td>
          <td>0.258377</td>
          <td>26.750105</td>
          <td>0.151654</td>
          <td>26.158677</td>
          <td>0.146804</td>
          <td>25.518489</td>
          <td>0.158953</td>
          <td>25.342588</td>
          <td>0.295131</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.675225</td>
          <td>0.431340</td>
          <td>26.524880</td>
          <td>0.141491</td>
          <td>26.080023</td>
          <td>0.084613</td>
          <td>25.700752</td>
          <td>0.098621</td>
          <td>25.373217</td>
          <td>0.140316</td>
          <td>25.012056</td>
          <td>0.225161</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.685880</td>
          <td>0.186485</td>
          <td>25.971593</td>
          <td>0.090439</td>
          <td>25.425939</td>
          <td>0.091697</td>
          <td>25.031973</td>
          <td>0.122447</td>
          <td>24.590722</td>
          <td>0.185474</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.892631</td>
          <td>2.667359</td>
          <td>28.035980</td>
          <td>0.542973</td>
          <td>27.019421</td>
          <td>0.222495</td>
          <td>28.069434</td>
          <td>0.753840</td>
          <td>28.574757</td>
          <td>1.535082</td>
          <td>26.813700</td>
          <td>0.969204</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.606210</td>
          <td>0.459536</td>
          <td>25.970639</td>
          <td>0.102785</td>
          <td>24.788345</td>
          <td>0.032475</td>
          <td>23.866999</td>
          <td>0.023741</td>
          <td>23.104490</td>
          <td>0.022979</td>
          <td>22.823775</td>
          <td>0.040436</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.750797</td>
          <td>1.597101</td>
          <td>27.629672</td>
          <td>0.386596</td>
          <td>27.079673</td>
          <td>0.390783</td>
          <td>26.324100</td>
          <td>0.380843</td>
          <td>25.207699</td>
          <td>0.328365</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.077615</td>
          <td>0.300178</td>
          <td>25.721256</td>
          <td>0.080962</td>
          <td>25.343946</td>
          <td>0.051930</td>
          <td>24.826343</td>
          <td>0.053987</td>
          <td>24.436828</td>
          <td>0.072678</td>
          <td>23.753183</td>
          <td>0.089960</td>
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
          <td>27.269376</td>
          <td>0.735518</td>
          <td>26.371274</td>
          <td>0.145292</td>
          <td>26.425904</td>
          <td>0.137205</td>
          <td>26.362745</td>
          <td>0.209739</td>
          <td>25.674547</td>
          <td>0.216118</td>
          <td>27.574146</td>
          <td>1.496889</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.086217</td>
          <td>0.642763</td>
          <td>27.348589</td>
          <td>0.322610</td>
          <td>26.908640</td>
          <td>0.203600</td>
          <td>26.753991</td>
          <td>0.284773</td>
          <td>25.837162</td>
          <td>0.243418</td>
          <td>25.591033</td>
          <td>0.418064</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.124205</td>
          <td>1.999965</td>
          <td>27.317388</td>
          <td>0.316916</td>
          <td>26.564362</td>
          <td>0.153315</td>
          <td>26.368610</td>
          <td>0.209074</td>
          <td>25.862293</td>
          <td>0.250514</td>
          <td>25.403311</td>
          <td>0.364417</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.063170</td>
          <td>0.262126</td>
          <td>26.400406</td>
          <td>0.135600</td>
          <td>25.843882</td>
          <td>0.136315</td>
          <td>25.912511</td>
          <td>0.265596</td>
          <td>25.104503</td>
          <td>0.292415</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.236512</td>
          <td>0.342912</td>
          <td>26.573967</td>
          <td>0.171108</td>
          <td>26.108068</td>
          <td>0.102965</td>
          <td>25.492167</td>
          <td>0.098199</td>
          <td>25.120818</td>
          <td>0.133555</td>
          <td>24.617669</td>
          <td>0.191629</td>
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
          <td>26.602627</td>
          <td>0.151279</td>
          <td>25.922866</td>
          <td>0.073662</td>
          <td>25.164288</td>
          <td>0.061429</td>
          <td>24.953197</td>
          <td>0.097381</td>
          <td>24.840291</td>
          <td>0.195056</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.673315</td>
          <td>1.524191</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.228763</td>
          <td>0.227424</td>
          <td>27.702334</td>
          <td>0.510152</td>
          <td>26.257979</td>
          <td>0.294470</td>
          <td>25.478302</td>
          <td>0.329269</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.600301</td>
          <td>0.863343</td>
          <td>25.817309</td>
          <td>0.082016</td>
          <td>24.783617</td>
          <td>0.029187</td>
          <td>23.858944</td>
          <td>0.021240</td>
          <td>23.111622</td>
          <td>0.020915</td>
          <td>22.858846</td>
          <td>0.037501</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.293552</td>
          <td>0.766292</td>
          <td>27.912325</td>
          <td>0.520417</td>
          <td>27.065162</td>
          <td>0.245286</td>
          <td>26.658578</td>
          <td>0.278923</td>
          <td>26.095066</td>
          <td>0.316984</td>
          <td>25.108600</td>
          <td>0.302372</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.002401</td>
          <td>0.253570</td>
          <td>25.703792</td>
          <td>0.069100</td>
          <td>25.490748</td>
          <td>0.050280</td>
          <td>24.909132</td>
          <td>0.049045</td>
          <td>24.375241</td>
          <td>0.058528</td>
          <td>23.625768</td>
          <td>0.068040</td>
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
          <td>26.175140</td>
          <td>0.306799</td>
          <td>26.393488</td>
          <td>0.135194</td>
          <td>26.000033</td>
          <td>0.085339</td>
          <td>26.540707</td>
          <td>0.219648</td>
          <td>25.688497</td>
          <td>0.198078</td>
          <td>25.800176</td>
          <td>0.453458</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.272792</td>
          <td>0.671135</td>
          <td>27.157796</td>
          <td>0.244665</td>
          <td>26.637404</td>
          <td>0.139884</td>
          <td>26.482117</td>
          <td>0.196569</td>
          <td>26.366609</td>
          <td>0.325797</td>
          <td>24.997241</td>
          <td>0.226033</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.425622</td>
          <td>0.757251</td>
          <td>26.827307</td>
          <td>0.190833</td>
          <td>26.793441</td>
          <td>0.165055</td>
          <td>26.365138</td>
          <td>0.183998</td>
          <td>26.185245</td>
          <td>0.290172</td>
          <td>25.354913</td>
          <td>0.312202</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.077080</td>
          <td>0.619538</td>
          <td>27.414294</td>
          <td>0.326116</td>
          <td>26.734002</td>
          <td>0.167234</td>
          <td>25.775580</td>
          <td>0.118598</td>
          <td>25.483021</td>
          <td>0.172347</td>
          <td>25.301049</td>
          <td>0.318095</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.840206</td>
          <td>0.499485</td>
          <td>26.555292</td>
          <td>0.150137</td>
          <td>26.205677</td>
          <td>0.098249</td>
          <td>25.593232</td>
          <td>0.093476</td>
          <td>25.234057</td>
          <td>0.129271</td>
          <td>25.216579</td>
          <td>0.276619</td>
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
