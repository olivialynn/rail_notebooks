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

    <pzflow.flow.Flow at 0x7f203c873d60>



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
    0      23.994413  0.079976  0.041551  
    1      25.391064  0.071346  0.047650  
    2      24.304707  0.087887  0.051794  
    3      25.291103  0.068797  0.041196  
    4      25.096743  0.182746  0.173391  
    ...          ...       ...       ...  
    99995  24.737946  0.010742  0.006636  
    99996  24.224169  0.088381  0.086691  
    99997  25.613836  0.114595  0.088431  
    99998  25.274899  0.043829  0.023706  
    99999  25.699642  0.131829  0.123464  
    
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
          <td>28.348952</td>
          <td>1.288471</td>
          <td>26.689263</td>
          <td>0.162894</td>
          <td>25.948299</td>
          <td>0.075327</td>
          <td>25.234577</td>
          <td>0.065370</td>
          <td>24.761460</td>
          <td>0.082259</td>
          <td>23.996069</td>
          <td>0.094187</td>
          <td>0.079976</td>
          <td>0.041551</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.689211</td>
          <td>0.369946</td>
          <td>26.787149</td>
          <td>0.156544</td>
          <td>26.416378</td>
          <td>0.182902</td>
          <td>26.060549</td>
          <td>0.250535</td>
          <td>25.323145</td>
          <td>0.290539</td>
          <td>0.071346</td>
          <td>0.047650</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.175697</td>
          <td>1.922298</td>
          <td>28.234645</td>
          <td>0.557246</td>
          <td>28.765728</td>
          <td>0.729575</td>
          <td>26.102194</td>
          <td>0.139838</td>
          <td>25.088656</td>
          <td>0.109620</td>
          <td>24.311386</td>
          <td>0.124039</td>
          <td>0.087887</td>
          <td>0.051794</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.589926</td>
          <td>1.461351</td>
          <td>29.350613</td>
          <td>1.146231</td>
          <td>27.404465</td>
          <td>0.262609</td>
          <td>26.177811</td>
          <td>0.149237</td>
          <td>25.337203</td>
          <td>0.136024</td>
          <td>25.452756</td>
          <td>0.322362</td>
          <td>0.068797</td>
          <td>0.041196</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.305039</td>
          <td>0.323461</td>
          <td>26.154348</td>
          <td>0.102581</td>
          <td>25.981745</td>
          <td>0.077586</td>
          <td>25.677385</td>
          <td>0.096620</td>
          <td>25.414708</td>
          <td>0.145418</td>
          <td>25.172870</td>
          <td>0.257110</td>
          <td>0.182746</td>
          <td>0.173391</td>
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
          <td>26.874483</td>
          <td>0.500684</td>
          <td>26.431855</td>
          <td>0.130580</td>
          <td>25.279248</td>
          <td>0.041615</td>
          <td>25.086992</td>
          <td>0.057349</td>
          <td>24.853747</td>
          <td>0.089224</td>
          <td>24.325924</td>
          <td>0.125613</td>
          <td>0.010742</td>
          <td>0.006636</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.526170</td>
          <td>0.384758</td>
          <td>26.366420</td>
          <td>0.123387</td>
          <td>26.016893</td>
          <td>0.080031</td>
          <td>25.200426</td>
          <td>0.063420</td>
          <td>24.813234</td>
          <td>0.086099</td>
          <td>24.219715</td>
          <td>0.114537</td>
          <td>0.088381</td>
          <td>0.086691</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.022770</td>
          <td>0.557788</td>
          <td>26.808765</td>
          <td>0.180306</td>
          <td>26.441949</td>
          <td>0.116189</td>
          <td>26.118713</td>
          <td>0.141842</td>
          <td>25.839158</td>
          <td>0.208508</td>
          <td>25.907484</td>
          <td>0.458442</td>
          <td>0.114595</td>
          <td>0.088431</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.435084</td>
          <td>0.358413</td>
          <td>26.323346</td>
          <td>0.118860</td>
          <td>26.021360</td>
          <td>0.080347</td>
          <td>26.060739</td>
          <td>0.134923</td>
          <td>25.577429</td>
          <td>0.167153</td>
          <td>25.483962</td>
          <td>0.330458</td>
          <td>0.043829</td>
          <td>0.023706</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.163217</td>
          <td>0.616355</td>
          <td>26.588257</td>
          <td>0.149410</td>
          <td>26.484101</td>
          <td>0.120528</td>
          <td>26.416776</td>
          <td>0.182963</td>
          <td>25.682202</td>
          <td>0.182706</td>
          <td>25.949057</td>
          <td>0.472934</td>
          <td>0.131829</td>
          <td>0.123464</td>
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
          <td>27.120683</td>
          <td>0.661991</td>
          <td>26.887905</td>
          <td>0.223463</td>
          <td>26.100757</td>
          <td>0.102668</td>
          <td>25.030739</td>
          <td>0.065616</td>
          <td>24.721393</td>
          <td>0.094633</td>
          <td>23.937794</td>
          <td>0.107201</td>
          <td>0.079976</td>
          <td>0.041551</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.180466</td>
          <td>0.283787</td>
          <td>26.600594</td>
          <td>0.158075</td>
          <td>26.051239</td>
          <td>0.159776</td>
          <td>25.770433</td>
          <td>0.232135</td>
          <td>25.755192</td>
          <td>0.476616</td>
          <td>0.071346</td>
          <td>0.047650</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.943865</td>
          <td>1.004038</td>
          <td>28.271393</td>
          <td>0.596940</td>
          <td>25.980835</td>
          <td>0.151188</td>
          <td>25.280601</td>
          <td>0.154332</td>
          <td>24.131211</td>
          <td>0.127329</td>
          <td>0.087887</td>
          <td>0.051794</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.536485</td>
          <td>0.870151</td>
          <td>29.539700</td>
          <td>1.395866</td>
          <td>26.953559</td>
          <td>0.212734</td>
          <td>26.167556</td>
          <td>0.176139</td>
          <td>25.961149</td>
          <td>0.271103</td>
          <td>25.486894</td>
          <td>0.388189</td>
          <td>0.068797</td>
          <td>0.041196</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.866991</td>
          <td>0.271438</td>
          <td>26.191494</td>
          <td>0.133266</td>
          <td>25.840645</td>
          <td>0.088874</td>
          <td>25.572334</td>
          <td>0.115213</td>
          <td>25.525144</td>
          <td>0.205126</td>
          <td>25.308550</td>
          <td>0.365542</td>
          <td>0.182746</td>
          <td>0.173391</td>
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
          <td>27.153223</td>
          <td>0.671615</td>
          <td>26.387558</td>
          <td>0.144659</td>
          <td>25.524009</td>
          <td>0.060919</td>
          <td>25.084596</td>
          <td>0.067869</td>
          <td>24.777000</td>
          <td>0.098050</td>
          <td>24.463140</td>
          <td>0.166481</td>
          <td>0.010742</td>
          <td>0.006636</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.691023</td>
          <td>0.191469</td>
          <td>26.118136</td>
          <td>0.105478</td>
          <td>25.293052</td>
          <td>0.083758</td>
          <td>24.851065</td>
          <td>0.107269</td>
          <td>24.700454</td>
          <td>0.208546</td>
          <td>0.088381</td>
          <td>0.086691</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.190002</td>
          <td>0.702764</td>
          <td>26.638230</td>
          <td>0.184561</td>
          <td>26.235028</td>
          <td>0.117835</td>
          <td>26.085481</td>
          <td>0.168186</td>
          <td>26.399439</td>
          <td>0.392233</td>
          <td>26.479644</td>
          <td>0.805293</td>
          <td>0.114595</td>
          <td>0.088431</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.514551</td>
          <td>0.423698</td>
          <td>26.085404</td>
          <td>0.111769</td>
          <td>26.233263</td>
          <td>0.114188</td>
          <td>25.528369</td>
          <td>0.100752</td>
          <td>25.807608</td>
          <td>0.237555</td>
          <td>24.944578</td>
          <td>0.250145</td>
          <td>0.043829</td>
          <td>0.023706</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.093120</td>
          <td>0.665031</td>
          <td>26.427427</td>
          <td>0.156780</td>
          <td>26.415230</td>
          <td>0.140227</td>
          <td>26.026177</td>
          <td>0.162841</td>
          <td>25.937667</td>
          <td>0.276450</td>
          <td>26.226876</td>
          <td>0.690243</td>
          <td>0.131829</td>
          <td>0.123464</td>
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
          <td>27.060953</td>
          <td>0.232460</td>
          <td>26.065208</td>
          <td>0.087936</td>
          <td>25.204666</td>
          <td>0.067217</td>
          <td>24.860541</td>
          <td>0.094494</td>
          <td>23.874723</td>
          <td>0.089301</td>
          <td>0.079976</td>
          <td>0.041551</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.566252</td>
          <td>0.829376</td>
          <td>27.459502</td>
          <td>0.320301</td>
          <td>26.494052</td>
          <td>0.127386</td>
          <td>26.228362</td>
          <td>0.163540</td>
          <td>26.185888</td>
          <td>0.289884</td>
          <td>25.497143</td>
          <td>0.348967</td>
          <td>0.071346</td>
          <td>0.047650</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.260126</td>
          <td>0.683528</td>
          <td>30.085266</td>
          <td>1.728344</td>
          <td>27.099040</td>
          <td>0.216927</td>
          <td>26.148053</td>
          <td>0.155467</td>
          <td>25.182067</td>
          <td>0.126801</td>
          <td>24.217379</td>
          <td>0.122195</td>
          <td>0.087887</td>
          <td>0.051794</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.532642</td>
          <td>0.642577</td>
          <td>26.100827</td>
          <td>0.145746</td>
          <td>25.609088</td>
          <td>0.178727</td>
          <td>24.848312</td>
          <td>0.204588</td>
          <td>0.068797</td>
          <td>0.041196</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.079781</td>
          <td>0.333539</td>
          <td>26.487420</td>
          <td>0.179242</td>
          <td>26.085986</td>
          <td>0.115579</td>
          <td>25.665625</td>
          <td>0.131193</td>
          <td>25.552594</td>
          <td>0.219716</td>
          <td>26.006019</td>
          <td>0.638737</td>
          <td>0.182746</td>
          <td>0.173391</td>
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
          <td>26.378968</td>
          <td>0.343185</td>
          <td>26.187809</td>
          <td>0.105723</td>
          <td>25.391120</td>
          <td>0.046008</td>
          <td>25.122556</td>
          <td>0.059256</td>
          <td>24.783250</td>
          <td>0.083946</td>
          <td>24.951805</td>
          <td>0.214371</td>
          <td>0.010742</td>
          <td>0.006636</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.524141</td>
          <td>0.153313</td>
          <td>26.150331</td>
          <td>0.098912</td>
          <td>25.182355</td>
          <td>0.068946</td>
          <td>25.015195</td>
          <td>0.112896</td>
          <td>24.257337</td>
          <td>0.130331</td>
          <td>0.088381</td>
          <td>0.086691</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.133002</td>
          <td>0.305763</td>
          <td>26.490344</td>
          <td>0.152831</td>
          <td>26.567446</td>
          <td>0.146321</td>
          <td>26.285580</td>
          <td>0.185443</td>
          <td>26.437021</td>
          <td>0.379000</td>
          <td>26.242814</td>
          <td>0.649039</td>
          <td>0.114595</td>
          <td>0.088431</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.771906</td>
          <td>0.211663</td>
          <td>26.148686</td>
          <td>0.103544</td>
          <td>25.975564</td>
          <td>0.078456</td>
          <td>25.823830</td>
          <td>0.111738</td>
          <td>26.023224</td>
          <td>0.246739</td>
          <td>24.904879</td>
          <td>0.209291</td>
          <td>0.043829</td>
          <td>0.023706</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.339168</td>
          <td>0.770182</td>
          <td>26.691038</td>
          <td>0.190658</td>
          <td>26.591927</td>
          <td>0.158160</td>
          <td>26.096310</td>
          <td>0.167407</td>
          <td>25.431754</td>
          <td>0.176101</td>
          <td>25.633499</td>
          <td>0.438099</td>
          <td>0.131829</td>
          <td>0.123464</td>
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
