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

    <pzflow.flow.Flow at 0x7f1d414824d0>



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
    0      23.994413  0.048589  0.026912  
    1      25.391064  0.018794  0.010902  
    2      24.304707  0.020591  0.019276  
    3      25.291103  0.145624  0.082107  
    4      25.096743  0.068101  0.062288  
    ...          ...       ...       ...  
    99995  24.737946  0.001532  0.001021  
    99996  24.224169  0.022389  0.016338  
    99997  25.613836  0.077486  0.065110  
    99998  25.274899  0.012699  0.009756  
    99999  25.699642  0.058731  0.056567  
    
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
          <td>26.850385</td>
          <td>0.491853</td>
          <td>26.531834</td>
          <td>0.142340</td>
          <td>25.986043</td>
          <td>0.077881</td>
          <td>25.175773</td>
          <td>0.062049</td>
          <td>24.669167</td>
          <td>0.075824</td>
          <td>24.072796</td>
          <td>0.100742</td>
          <td>0.048589</td>
          <td>0.026912</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.068584</td>
          <td>0.224219</td>
          <td>26.646515</td>
          <td>0.138726</td>
          <td>26.397208</td>
          <td>0.179956</td>
          <td>25.476217</td>
          <td>0.153303</td>
          <td>25.312670</td>
          <td>0.288091</td>
          <td>0.018794</td>
          <td>0.010902</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.378998</td>
          <td>0.617491</td>
          <td>29.092008</td>
          <td>0.901419</td>
          <td>26.186791</td>
          <td>0.150392</td>
          <td>25.084760</td>
          <td>0.109248</td>
          <td>24.423879</td>
          <td>0.136722</td>
          <td>0.020591</td>
          <td>0.019276</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.578634</td>
          <td>0.339184</td>
          <td>27.366386</td>
          <td>0.254548</td>
          <td>26.042931</td>
          <td>0.132863</td>
          <td>25.652713</td>
          <td>0.178199</td>
          <td>25.378911</td>
          <td>0.303880</td>
          <td>0.145624</td>
          <td>0.082107</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.030655</td>
          <td>0.560959</td>
          <td>26.145621</td>
          <td>0.101801</td>
          <td>26.121069</td>
          <td>0.087727</td>
          <td>25.565445</td>
          <td>0.087568</td>
          <td>25.481344</td>
          <td>0.153979</td>
          <td>25.280025</td>
          <td>0.280577</td>
          <td>0.068101</td>
          <td>0.062288</td>
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
          <td>26.953082</td>
          <td>0.530354</td>
          <td>26.644876</td>
          <td>0.156834</td>
          <td>25.474384</td>
          <td>0.049484</td>
          <td>25.108135</td>
          <td>0.058435</td>
          <td>24.810517</td>
          <td>0.085893</td>
          <td>24.732085</td>
          <td>0.177991</td>
          <td>0.001532</td>
          <td>0.001021</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.550876</td>
          <td>0.392177</td>
          <td>26.602551</td>
          <td>0.151253</td>
          <td>26.076706</td>
          <td>0.084366</td>
          <td>25.272272</td>
          <td>0.067590</td>
          <td>24.892468</td>
          <td>0.092314</td>
          <td>24.379645</td>
          <td>0.131595</td>
          <td>0.022389</td>
          <td>0.016338</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.906092</td>
          <td>0.512455</td>
          <td>26.711257</td>
          <td>0.165977</td>
          <td>26.438043</td>
          <td>0.115795</td>
          <td>26.245594</td>
          <td>0.158163</td>
          <td>25.597937</td>
          <td>0.170098</td>
          <td>26.557109</td>
          <td>0.728148</td>
          <td>0.077486</td>
          <td>0.065110</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.056942</td>
          <td>0.264884</td>
          <td>26.224709</td>
          <td>0.109081</td>
          <td>26.000393</td>
          <td>0.078874</td>
          <td>25.933296</td>
          <td>0.120818</td>
          <td>25.921805</td>
          <td>0.223390</td>
          <td>25.177662</td>
          <td>0.258121</td>
          <td>0.012699</td>
          <td>0.009756</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.953305</td>
          <td>0.243329</td>
          <td>26.767127</td>
          <td>0.174053</td>
          <td>26.449737</td>
          <td>0.116979</td>
          <td>26.305018</td>
          <td>0.166395</td>
          <td>25.679555</td>
          <td>0.182297</td>
          <td>25.441550</td>
          <td>0.319496</td>
          <td>0.058731</td>
          <td>0.056567</td>
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
          <td>26.522633</td>
          <td>0.163099</td>
          <td>26.095618</td>
          <td>0.101358</td>
          <td>25.383488</td>
          <td>0.088815</td>
          <td>24.754349</td>
          <td>0.096599</td>
          <td>23.991359</td>
          <td>0.111375</td>
          <td>0.048589</td>
          <td>0.026912</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.187200</td>
          <td>0.282609</td>
          <td>26.473439</td>
          <td>0.140135</td>
          <td>26.340043</td>
          <td>0.201724</td>
          <td>25.585679</td>
          <td>0.196790</td>
          <td>25.295043</td>
          <td>0.330951</td>
          <td>0.018794</td>
          <td>0.010902</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.982462</td>
          <td>1.873808</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.487695</td>
          <td>1.265934</td>
          <td>26.015497</td>
          <td>0.153253</td>
          <td>25.280924</td>
          <td>0.151987</td>
          <td>24.354805</td>
          <td>0.151923</td>
          <td>0.020591</td>
          <td>0.019276</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.809354</td>
          <td>0.194736</td>
          <td>26.148084</td>
          <td>0.179241</td>
          <td>25.416661</td>
          <td>0.177994</td>
          <td>25.444255</td>
          <td>0.387325</td>
          <td>0.145624</td>
          <td>0.082107</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.884755</td>
          <td>0.560941</td>
          <td>26.395933</td>
          <td>0.147520</td>
          <td>25.932919</td>
          <td>0.088674</td>
          <td>25.800757</td>
          <td>0.129056</td>
          <td>25.557155</td>
          <td>0.194621</td>
          <td>25.225127</td>
          <td>0.316997</td>
          <td>0.068101</td>
          <td>0.062288</td>
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
          <td>26.274097</td>
          <td>0.131155</td>
          <td>25.407174</td>
          <td>0.054907</td>
          <td>24.996923</td>
          <td>0.062781</td>
          <td>25.083927</td>
          <td>0.128086</td>
          <td>24.625640</td>
          <td>0.191016</td>
          <td>0.001532</td>
          <td>0.001021</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.234822</td>
          <td>1.302655</td>
          <td>27.082366</td>
          <td>0.259606</td>
          <td>26.006421</td>
          <td>0.093368</td>
          <td>25.238209</td>
          <td>0.077826</td>
          <td>24.829798</td>
          <td>0.102796</td>
          <td>24.279627</td>
          <td>0.142414</td>
          <td>0.022389</td>
          <td>0.016338</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.273507</td>
          <td>0.735882</td>
          <td>26.519559</td>
          <td>0.164382</td>
          <td>26.318768</td>
          <td>0.124583</td>
          <td>26.075333</td>
          <td>0.163883</td>
          <td>26.159705</td>
          <td>0.319988</td>
          <td>25.824700</td>
          <td>0.503881</td>
          <td>0.077486</td>
          <td>0.065110</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.949058</td>
          <td>0.270582</td>
          <td>26.337475</td>
          <td>0.138579</td>
          <td>26.245967</td>
          <td>0.115027</td>
          <td>25.668629</td>
          <td>0.113442</td>
          <td>25.519701</td>
          <td>0.186079</td>
          <td>25.038086</td>
          <td>0.269064</td>
          <td>0.012699</td>
          <td>0.009756</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.775774</td>
          <td>0.517306</td>
          <td>26.493122</td>
          <td>0.159881</td>
          <td>26.746518</td>
          <td>0.178785</td>
          <td>26.361498</td>
          <td>0.207503</td>
          <td>25.523550</td>
          <td>0.188625</td>
          <td>27.394595</td>
          <td>1.357255</td>
          <td>0.058731</td>
          <td>0.056567</td>
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
          <td>29.092704</td>
          <td>1.867163</td>
          <td>26.744088</td>
          <td>0.173635</td>
          <td>26.234872</td>
          <td>0.098950</td>
          <td>25.138721</td>
          <td>0.061357</td>
          <td>24.605026</td>
          <td>0.073131</td>
          <td>24.000549</td>
          <td>0.096582</td>
          <td>0.048589</td>
          <td>0.026912</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.984708</td>
          <td>3.572291</td>
          <td>27.789997</td>
          <td>0.400966</td>
          <td>26.810375</td>
          <td>0.160183</td>
          <td>26.451285</td>
          <td>0.188987</td>
          <td>26.297009</td>
          <td>0.304486</td>
          <td>25.916593</td>
          <td>0.462895</td>
          <td>0.018794</td>
          <td>0.010902</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.817404</td>
          <td>0.481484</td>
          <td>28.096920</td>
          <td>0.506036</td>
          <td>28.687586</td>
          <td>0.694936</td>
          <td>25.984512</td>
          <td>0.127016</td>
          <td>25.125262</td>
          <td>0.113781</td>
          <td>24.484212</td>
          <td>0.144809</td>
          <td>0.020591</td>
          <td>0.019276</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.801747</td>
          <td>0.413674</td>
          <td>26.088440</td>
          <td>0.161751</td>
          <td>25.623814</td>
          <td>0.201724</td>
          <td>25.202528</td>
          <td>0.305186</td>
          <td>0.145624</td>
          <td>0.082107</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.753922</td>
          <td>0.213924</td>
          <td>26.079174</td>
          <td>0.100733</td>
          <td>25.868651</td>
          <td>0.074178</td>
          <td>25.787011</td>
          <td>0.112582</td>
          <td>25.260110</td>
          <td>0.134277</td>
          <td>24.755404</td>
          <td>0.191708</td>
          <td>0.068101</td>
          <td>0.062288</td>
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
          <td>26.771498</td>
          <td>0.463815</td>
          <td>26.187339</td>
          <td>0.105583</td>
          <td>25.460556</td>
          <td>0.048881</td>
          <td>25.004445</td>
          <td>0.053298</td>
          <td>24.737054</td>
          <td>0.080509</td>
          <td>24.928776</td>
          <td>0.210066</td>
          <td>0.001532</td>
          <td>0.001021</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.699430</td>
          <td>0.165032</td>
          <td>26.020538</td>
          <td>0.080709</td>
          <td>25.156222</td>
          <td>0.061319</td>
          <td>24.891641</td>
          <td>0.092727</td>
          <td>24.126957</td>
          <td>0.106200</td>
          <td>0.022389</td>
          <td>0.016338</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.385535</td>
          <td>0.359154</td>
          <td>26.705126</td>
          <td>0.174438</td>
          <td>26.617600</td>
          <td>0.144203</td>
          <td>26.467616</td>
          <td>0.203819</td>
          <td>26.374143</td>
          <td>0.342346</td>
          <td>25.530534</td>
          <td>0.364109</td>
          <td>0.077486</td>
          <td>0.065110</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.750229</td>
          <td>0.205840</td>
          <td>26.204389</td>
          <td>0.107325</td>
          <td>26.131078</td>
          <td>0.088658</td>
          <td>26.030849</td>
          <td>0.131719</td>
          <td>25.842350</td>
          <td>0.209412</td>
          <td>25.442457</td>
          <td>0.320252</td>
          <td>0.012699</td>
          <td>0.009756</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.127586</td>
          <td>0.243928</td>
          <td>26.439779</td>
          <td>0.121037</td>
          <td>26.342043</td>
          <td>0.179400</td>
          <td>25.708724</td>
          <td>0.194746</td>
          <td>25.528488</td>
          <td>0.356323</td>
          <td>0.058731</td>
          <td>0.056567</td>
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
