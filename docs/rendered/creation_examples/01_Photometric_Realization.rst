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

    <pzflow.flow.Flow at 0x7fb2cae9e650>



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
    0      23.994413  0.074559  0.071514  
    1      25.391064  0.016524  0.013800  
    2      24.304707  0.076574  0.068200  
    3      25.291103  0.120152  0.079066  
    4      25.096743  0.043432  0.032676  
    ...          ...       ...       ...  
    99995  24.737946  0.068214  0.067383  
    99996  24.224169  0.178751  0.134740  
    99997  25.613836  0.006496  0.004050  
    99998  25.274899  0.014919  0.010639  
    99999  25.699642  0.006106  0.003230  
    
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
          <td>28.294223</td>
          <td>1.250752</td>
          <td>27.087906</td>
          <td>0.227844</td>
          <td>25.950559</td>
          <td>0.075477</td>
          <td>25.208431</td>
          <td>0.063872</td>
          <td>24.652034</td>
          <td>0.074684</td>
          <td>24.014722</td>
          <td>0.095741</td>
          <td>0.074559</td>
          <td>0.071514</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.046358</td>
          <td>0.567316</td>
          <td>27.706169</td>
          <td>0.374865</td>
          <td>26.563250</td>
          <td>0.129094</td>
          <td>26.176207</td>
          <td>0.149032</td>
          <td>25.791499</td>
          <td>0.200342</td>
          <td>24.953636</td>
          <td>0.214470</td>
          <td>0.016524</td>
          <td>0.013800</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.532640</td>
          <td>2.938824</td>
          <td>28.786714</td>
          <td>0.739892</td>
          <td>25.923085</td>
          <td>0.119751</td>
          <td>24.838374</td>
          <td>0.088026</td>
          <td>24.207131</td>
          <td>0.113288</td>
          <td>0.076574</td>
          <td>0.068200</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.551064</td>
          <td>0.295786</td>
          <td>26.292513</td>
          <td>0.164631</td>
          <td>25.495644</td>
          <td>0.155876</td>
          <td>25.093794</td>
          <td>0.240926</td>
          <td>0.120152</td>
          <td>0.079066</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.033864</td>
          <td>0.259942</td>
          <td>26.202656</td>
          <td>0.107002</td>
          <td>25.925453</td>
          <td>0.073821</td>
          <td>25.621977</td>
          <td>0.092032</td>
          <td>25.523620</td>
          <td>0.159651</td>
          <td>25.297851</td>
          <td>0.284658</td>
          <td>0.043432</td>
          <td>0.032676</td>
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
          <td>26.367076</td>
          <td>0.339753</td>
          <td>26.340524</td>
          <td>0.120646</td>
          <td>25.335427</td>
          <td>0.043741</td>
          <td>25.115849</td>
          <td>0.058836</td>
          <td>24.901049</td>
          <td>0.093012</td>
          <td>24.596797</td>
          <td>0.158622</td>
          <td>0.068214</td>
          <td>0.067383</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.648125</td>
          <td>0.853112</td>
          <td>27.129794</td>
          <td>0.235885</td>
          <td>26.097856</td>
          <td>0.085952</td>
          <td>25.163437</td>
          <td>0.061374</td>
          <td>24.840109</td>
          <td>0.088160</td>
          <td>24.268972</td>
          <td>0.119553</td>
          <td>0.178751</td>
          <td>0.134740</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.291987</td>
          <td>0.320120</td>
          <td>26.607359</td>
          <td>0.151877</td>
          <td>26.162282</td>
          <td>0.090966</td>
          <td>26.320309</td>
          <td>0.168577</td>
          <td>25.871254</td>
          <td>0.214177</td>
          <td>25.142063</td>
          <td>0.250693</td>
          <td>0.006496</td>
          <td>0.004050</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.989576</td>
          <td>0.250688</td>
          <td>26.132229</td>
          <td>0.100615</td>
          <td>26.152556</td>
          <td>0.090191</td>
          <td>25.943499</td>
          <td>0.121894</td>
          <td>25.862688</td>
          <td>0.212651</td>
          <td>24.839838</td>
          <td>0.194956</td>
          <td>0.014919</td>
          <td>0.010639</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.294456</td>
          <td>0.675109</td>
          <td>26.815505</td>
          <td>0.181338</td>
          <td>26.594043</td>
          <td>0.132580</td>
          <td>26.650306</td>
          <td>0.222574</td>
          <td>25.631223</td>
          <td>0.174979</td>
          <td>25.488809</td>
          <td>0.331731</td>
          <td>0.006106</td>
          <td>0.003230</td>
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
          <td>26.469109</td>
          <td>0.157554</td>
          <td>26.066786</td>
          <td>0.100085</td>
          <td>25.289089</td>
          <td>0.082811</td>
          <td>24.617269</td>
          <td>0.086729</td>
          <td>24.053666</td>
          <td>0.119106</td>
          <td>0.074559</td>
          <td>0.071514</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.655736</td>
          <td>0.830406</td>
          <td>26.939033</td>
          <td>0.208172</td>
          <td>26.531151</td>
          <td>0.236531</td>
          <td>25.898627</td>
          <td>0.255216</td>
          <td>25.471325</td>
          <td>0.380066</td>
          <td>0.016524</td>
          <td>0.013800</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.555628</td>
          <td>0.884015</td>
          <td>28.526916</td>
          <td>0.772211</td>
          <td>29.604226</td>
          <td>1.360621</td>
          <td>26.095294</td>
          <td>0.166771</td>
          <td>24.814501</td>
          <td>0.103088</td>
          <td>24.075478</td>
          <td>0.121352</td>
          <td>0.076574</td>
          <td>0.068200</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.826300</td>
          <td>0.436594</td>
          <td>26.177862</td>
          <td>0.181812</td>
          <td>25.554224</td>
          <td>0.197806</td>
          <td>25.045321</td>
          <td>0.279377</td>
          <td>0.120152</td>
          <td>0.079066</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.420439</td>
          <td>0.394431</td>
          <td>26.129570</td>
          <td>0.116230</td>
          <td>25.957796</td>
          <td>0.089795</td>
          <td>25.754359</td>
          <td>0.122797</td>
          <td>25.482034</td>
          <td>0.181039</td>
          <td>25.058488</td>
          <td>0.274761</td>
          <td>0.043432</td>
          <td>0.032676</td>
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
          <td>26.630601</td>
          <td>0.465886</td>
          <td>26.355722</td>
          <td>0.142652</td>
          <td>25.353621</td>
          <td>0.053184</td>
          <td>25.045725</td>
          <td>0.066620</td>
          <td>24.863072</td>
          <td>0.107326</td>
          <td>24.402770</td>
          <td>0.160533</td>
          <td>0.068214</td>
          <td>0.067383</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.843474</td>
          <td>1.815521</td>
          <td>26.487279</td>
          <td>0.168777</td>
          <td>25.988849</td>
          <td>0.099281</td>
          <td>25.179269</td>
          <td>0.080021</td>
          <td>24.881141</td>
          <td>0.116035</td>
          <td>24.271709</td>
          <td>0.152784</td>
          <td>0.178751</td>
          <td>0.134740</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.343874</td>
          <td>0.370489</td>
          <td>26.523164</td>
          <td>0.162443</td>
          <td>26.252228</td>
          <td>0.115618</td>
          <td>26.598184</td>
          <td>0.249800</td>
          <td>26.157554</td>
          <td>0.314540</td>
          <td>25.973096</td>
          <td>0.553356</td>
          <td>0.006496</td>
          <td>0.004050</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.143562</td>
          <td>0.316485</td>
          <td>26.258752</td>
          <td>0.129492</td>
          <td>25.916872</td>
          <td>0.086235</td>
          <td>25.819090</td>
          <td>0.129296</td>
          <td>25.825490</td>
          <td>0.240267</td>
          <td>25.832699</td>
          <td>0.499679</td>
          <td>0.014919</td>
          <td>0.010639</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.519206</td>
          <td>0.424047</td>
          <td>26.598730</td>
          <td>0.173228</td>
          <td>26.505324</td>
          <td>0.143934</td>
          <td>26.699403</td>
          <td>0.271358</td>
          <td>25.675140</td>
          <td>0.211969</td>
          <td>26.090809</td>
          <td>0.601872</td>
          <td>0.006106</td>
          <td>0.003230</td>
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
          <td>27.155685</td>
          <td>0.636838</td>
          <td>26.559298</td>
          <td>0.154398</td>
          <td>26.112776</td>
          <td>0.093186</td>
          <td>25.301512</td>
          <td>0.074489</td>
          <td>24.610495</td>
          <td>0.077055</td>
          <td>23.782049</td>
          <td>0.083698</td>
          <td>0.074559</td>
          <td>0.071514</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.708751</td>
          <td>0.376525</td>
          <td>26.792582</td>
          <td>0.157756</td>
          <td>25.959083</td>
          <td>0.123957</td>
          <td>25.487525</td>
          <td>0.155270</td>
          <td>24.842624</td>
          <td>0.196026</td>
          <td>0.016524</td>
          <td>0.013800</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.518040</td>
          <td>0.812118</td>
          <td>28.509023</td>
          <td>0.706942</td>
          <td>28.042555</td>
          <td>0.460181</td>
          <td>26.301933</td>
          <td>0.177535</td>
          <td>25.170331</td>
          <td>0.125706</td>
          <td>24.395051</td>
          <td>0.142711</td>
          <td>0.076574</td>
          <td>0.068200</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.249414</td>
          <td>2.965827</td>
          <td>28.334013</td>
          <td>0.651406</td>
          <td>27.173911</td>
          <td>0.243403</td>
          <td>26.487853</td>
          <td>0.219193</td>
          <td>25.881597</td>
          <td>0.242209</td>
          <td>24.704597</td>
          <td>0.196200</td>
          <td>0.120152</td>
          <td>0.079066</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.068630</td>
          <td>0.270831</td>
          <td>26.300872</td>
          <td>0.118548</td>
          <td>25.973453</td>
          <td>0.078557</td>
          <td>25.548102</td>
          <td>0.088042</td>
          <td>25.239707</td>
          <td>0.127462</td>
          <td>24.924274</td>
          <td>0.213371</td>
          <td>0.043432</td>
          <td>0.032676</td>
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
          <td>27.098864</td>
          <td>0.608915</td>
          <td>26.355309</td>
          <td>0.128553</td>
          <td>25.447480</td>
          <td>0.051279</td>
          <td>25.140837</td>
          <td>0.064017</td>
          <td>24.943136</td>
          <td>0.102334</td>
          <td>25.051985</td>
          <td>0.246446</td>
          <td>0.068214</td>
          <td>0.067383</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.683511</td>
          <td>0.509878</td>
          <td>26.714485</td>
          <td>0.207066</td>
          <td>26.100363</td>
          <td>0.111010</td>
          <td>25.119866</td>
          <td>0.077060</td>
          <td>24.526856</td>
          <td>0.086314</td>
          <td>24.087355</td>
          <td>0.132238</td>
          <td>0.178751</td>
          <td>0.134740</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.633265</td>
          <td>0.417881</td>
          <td>26.616021</td>
          <td>0.153060</td>
          <td>26.468346</td>
          <td>0.118935</td>
          <td>26.203982</td>
          <td>0.152689</td>
          <td>25.399803</td>
          <td>0.143621</td>
          <td>25.287600</td>
          <td>0.282412</td>
          <td>0.006496</td>
          <td>0.004050</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.013919</td>
          <td>1.068263</td>
          <td>26.140228</td>
          <td>0.101521</td>
          <td>25.944479</td>
          <td>0.075245</td>
          <td>26.093940</td>
          <td>0.139172</td>
          <td>25.651376</td>
          <td>0.178388</td>
          <td>25.435379</td>
          <td>0.318611</td>
          <td>0.014919</td>
          <td>0.010639</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.866257</td>
          <td>0.977304</td>
          <td>27.346929</td>
          <td>0.281839</td>
          <td>26.629043</td>
          <td>0.136694</td>
          <td>26.292288</td>
          <td>0.164654</td>
          <td>26.253580</td>
          <td>0.293261</td>
          <td>25.561944</td>
          <td>0.351560</td>
          <td>0.006106</td>
          <td>0.003230</td>
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
