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

    <pzflow.flow.Flow at 0x7fdffc40d000>



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
    0      23.994413  0.087550  0.054995  
    1      25.391064  0.004168  0.002650  
    2      24.304707  0.157480  0.142505  
    3      25.291103  0.030859  0.019919  
    4      25.096743  0.048816  0.040369  
    ...          ...       ...       ...  
    99995  24.737946  0.012947  0.012571  
    99996  24.224169  0.079165  0.075490  
    99997  25.613836  0.076763  0.052448  
    99998  25.274899  0.145889  0.111691  
    99999  25.699642  0.114736  0.098051  
    
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
          <td>27.575008</td>
          <td>0.813955</td>
          <td>26.882299</td>
          <td>0.191859</td>
          <td>25.985338</td>
          <td>0.077832</td>
          <td>25.146817</td>
          <td>0.060476</td>
          <td>24.666881</td>
          <td>0.075671</td>
          <td>24.044439</td>
          <td>0.098270</td>
          <td>0.087550</td>
          <td>0.054995</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.643928</td>
          <td>0.421193</td>
          <td>30.311076</td>
          <td>1.858498</td>
          <td>26.522818</td>
          <td>0.124649</td>
          <td>26.225698</td>
          <td>0.155493</td>
          <td>25.598035</td>
          <td>0.170112</td>
          <td>24.955478</td>
          <td>0.214799</td>
          <td>0.004168</td>
          <td>0.002650</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.081953</td>
          <td>0.447787</td>
          <td>25.858493</td>
          <td>0.113203</td>
          <td>24.888100</td>
          <td>0.091960</td>
          <td>24.162875</td>
          <td>0.108998</td>
          <td>0.157480</td>
          <td>0.142505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.833951</td>
          <td>0.838420</td>
          <td>27.321650</td>
          <td>0.245359</td>
          <td>26.305571</td>
          <td>0.166474</td>
          <td>25.417820</td>
          <td>0.145808</td>
          <td>25.137174</td>
          <td>0.249688</td>
          <td>0.030859</td>
          <td>0.019919</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.979143</td>
          <td>0.540488</td>
          <td>26.227463</td>
          <td>0.109343</td>
          <td>25.960383</td>
          <td>0.076135</td>
          <td>25.753916</td>
          <td>0.103322</td>
          <td>25.284889</td>
          <td>0.130009</td>
          <td>25.099977</td>
          <td>0.242157</td>
          <td>0.048816</td>
          <td>0.040369</td>
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
          <td>27.228600</td>
          <td>0.645136</td>
          <td>26.284162</td>
          <td>0.114879</td>
          <td>25.471700</td>
          <td>0.049366</td>
          <td>25.266381</td>
          <td>0.067238</td>
          <td>24.879343</td>
          <td>0.091255</td>
          <td>25.017640</td>
          <td>0.226207</td>
          <td>0.012947</td>
          <td>0.012571</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.319937</td>
          <td>1.268400</td>
          <td>26.754048</td>
          <td>0.172130</td>
          <td>26.124935</td>
          <td>0.088026</td>
          <td>25.347025</td>
          <td>0.072214</td>
          <td>24.892853</td>
          <td>0.092345</td>
          <td>24.070259</td>
          <td>0.100518</td>
          <td>0.079165</td>
          <td>0.075490</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.488887</td>
          <td>2.188122</td>
          <td>26.708197</td>
          <td>0.165545</td>
          <td>26.376925</td>
          <td>0.109787</td>
          <td>26.082490</td>
          <td>0.137481</td>
          <td>25.775284</td>
          <td>0.197631</td>
          <td>25.629842</td>
          <td>0.370646</td>
          <td>0.076763</td>
          <td>0.052448</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.162118</td>
          <td>0.288472</td>
          <td>26.365755</td>
          <td>0.123316</td>
          <td>26.029141</td>
          <td>0.080901</td>
          <td>25.887940</td>
          <td>0.116145</td>
          <td>26.317239</td>
          <td>0.308565</td>
          <td>25.652685</td>
          <td>0.377298</td>
          <td>0.145889</td>
          <td>0.111691</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.566613</td>
          <td>0.396964</td>
          <td>27.054160</td>
          <td>0.221547</td>
          <td>26.496213</td>
          <td>0.121803</td>
          <td>26.211046</td>
          <td>0.153553</td>
          <td>26.018236</td>
          <td>0.241959</td>
          <td>25.449233</td>
          <td>0.321458</td>
          <td>0.114736</td>
          <td>0.098051</td>
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
          <td>26.658476</td>
          <td>0.185056</td>
          <td>26.137818</td>
          <td>0.106497</td>
          <td>25.233615</td>
          <td>0.078855</td>
          <td>24.809867</td>
          <td>0.102694</td>
          <td>24.073339</td>
          <td>0.121154</td>
          <td>0.087550</td>
          <td>0.054995</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.891196</td>
          <td>0.488238</td>
          <td>26.587247</td>
          <td>0.154414</td>
          <td>25.872399</td>
          <td>0.135322</td>
          <td>25.764173</td>
          <td>0.228268</td>
          <td>25.818174</td>
          <td>0.494119</td>
          <td>0.004168</td>
          <td>0.002650</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.473072</td>
          <td>0.864549</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.699011</td>
          <td>0.409276</td>
          <td>26.099603</td>
          <td>0.176599</td>
          <td>24.817406</td>
          <td>0.109016</td>
          <td>24.518201</td>
          <td>0.187148</td>
          <td>0.157480</td>
          <td>0.142505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.697679</td>
          <td>0.422858</td>
          <td>27.356175</td>
          <td>0.293755</td>
          <td>26.066407</td>
          <td>0.160228</td>
          <td>25.220891</td>
          <td>0.144482</td>
          <td>26.791995</td>
          <td>0.957850</td>
          <td>0.030859</td>
          <td>0.019919</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.115089</td>
          <td>0.656759</td>
          <td>26.136800</td>
          <td>0.117149</td>
          <td>26.059473</td>
          <td>0.098353</td>
          <td>25.852145</td>
          <td>0.133892</td>
          <td>25.222556</td>
          <td>0.145327</td>
          <td>25.137152</td>
          <td>0.293324</td>
          <td>0.048816</td>
          <td>0.040369</td>
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
          <td>27.333056</td>
          <td>0.758333</td>
          <td>26.133634</td>
          <td>0.116178</td>
          <td>25.398989</td>
          <td>0.054540</td>
          <td>25.112211</td>
          <td>0.069569</td>
          <td>25.019722</td>
          <td>0.121214</td>
          <td>24.510602</td>
          <td>0.173390</td>
          <td>0.012947</td>
          <td>0.012571</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.663771</td>
          <td>0.186232</td>
          <td>26.036342</td>
          <td>0.097657</td>
          <td>25.170421</td>
          <td>0.074743</td>
          <td>24.911038</td>
          <td>0.112419</td>
          <td>24.481130</td>
          <td>0.172397</td>
          <td>0.079165</td>
          <td>0.075490</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.461736</td>
          <td>0.831308</td>
          <td>26.437488</td>
          <td>0.152895</td>
          <td>26.822038</td>
          <td>0.191173</td>
          <td>26.330652</td>
          <td>0.202864</td>
          <td>26.071750</td>
          <td>0.297500</td>
          <td>25.958028</td>
          <td>0.554050</td>
          <td>0.076763</td>
          <td>0.052448</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.489076</td>
          <td>0.429644</td>
          <td>26.011056</td>
          <td>0.109580</td>
          <td>26.041064</td>
          <td>0.101464</td>
          <td>26.091416</td>
          <td>0.172411</td>
          <td>25.848201</td>
          <td>0.257365</td>
          <td>25.298799</td>
          <td>0.348677</td>
          <td>0.145889</td>
          <td>0.111691</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.082642</td>
          <td>0.309276</td>
          <td>26.664541</td>
          <td>0.189187</td>
          <td>26.629113</td>
          <td>0.165943</td>
          <td>26.310552</td>
          <td>0.204013</td>
          <td>26.141963</td>
          <td>0.321309</td>
          <td>25.668530</td>
          <td>0.456525</td>
          <td>0.114736</td>
          <td>0.098051</td>
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
          <td>27.767445</td>
          <td>0.950731</td>
          <td>26.852829</td>
          <td>0.197955</td>
          <td>25.975074</td>
          <td>0.082494</td>
          <td>25.363169</td>
          <td>0.078600</td>
          <td>24.716548</td>
          <td>0.084552</td>
          <td>23.991813</td>
          <td>0.100561</td>
          <td>0.087550</td>
          <td>0.054995</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.606810</td>
          <td>0.346853</td>
          <td>26.672635</td>
          <td>0.141907</td>
          <td>26.172082</td>
          <td>0.148530</td>
          <td>26.299763</td>
          <td>0.304318</td>
          <td>25.664016</td>
          <td>0.380693</td>
          <td>0.004168</td>
          <td>0.002650</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.720915</td>
          <td>0.517856</td>
          <td>27.672005</td>
          <td>0.439037</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.327660</td>
          <td>0.214748</td>
          <td>24.891220</td>
          <td>0.116687</td>
          <td>24.262356</td>
          <td>0.151081</td>
          <td>0.157480</td>
          <td>0.142505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.024680</td>
          <td>1.804332</td>
          <td>27.884118</td>
          <td>0.432788</td>
          <td>27.750445</td>
          <td>0.349551</td>
          <td>26.249646</td>
          <td>0.160190</td>
          <td>25.582662</td>
          <td>0.169380</td>
          <td>25.640759</td>
          <td>0.376944</td>
          <td>0.030859</td>
          <td>0.019919</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.845319</td>
          <td>0.226480</td>
          <td>26.112968</td>
          <td>0.101236</td>
          <td>26.039486</td>
          <td>0.083845</td>
          <td>25.601846</td>
          <td>0.092970</td>
          <td>25.454114</td>
          <td>0.154362</td>
          <td>25.209637</td>
          <td>0.271808</td>
          <td>0.048816</td>
          <td>0.040369</td>
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
          <td>29.901964</td>
          <td>2.555955</td>
          <td>26.626480</td>
          <td>0.154674</td>
          <td>25.405142</td>
          <td>0.046638</td>
          <td>25.081649</td>
          <td>0.057212</td>
          <td>24.767937</td>
          <td>0.082914</td>
          <td>24.873264</td>
          <td>0.200953</td>
          <td>0.012947</td>
          <td>0.012571</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.715198</td>
          <td>0.177472</td>
          <td>26.022472</td>
          <td>0.086727</td>
          <td>25.161071</td>
          <td>0.066310</td>
          <td>24.747938</td>
          <td>0.087643</td>
          <td>24.346367</td>
          <td>0.138040</td>
          <td>0.079165</td>
          <td>0.075490</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.941780</td>
          <td>0.543158</td>
          <td>26.711502</td>
          <td>0.173925</td>
          <td>26.492126</td>
          <td>0.128151</td>
          <td>26.482848</td>
          <td>0.204420</td>
          <td>25.795103</td>
          <td>0.211758</td>
          <td>26.110899</td>
          <td>0.558476</td>
          <td>0.076763</td>
          <td>0.052448</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.560624</td>
          <td>0.199762</td>
          <td>26.026609</td>
          <td>0.108122</td>
          <td>26.035046</td>
          <td>0.097955</td>
          <td>25.893271</td>
          <td>0.141189</td>
          <td>25.814177</td>
          <td>0.243358</td>
          <td>25.299009</td>
          <td>0.339229</td>
          <td>0.145889</td>
          <td>0.111691</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.279930</td>
          <td>0.345883</td>
          <td>26.445932</td>
          <td>0.148398</td>
          <td>26.496715</td>
          <td>0.139024</td>
          <td>26.736457</td>
          <td>0.272291</td>
          <td>25.812459</td>
          <td>0.231448</td>
          <td>27.220875</td>
          <td>1.205467</td>
          <td>0.114736</td>
          <td>0.098051</td>
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
