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

    <pzflow.flow.Flow at 0x7f86708ee9e0>



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
    0      23.994413  0.161398  0.120450  
    1      25.391064  0.082764  0.079149  
    2      24.304707  0.150925  0.120204  
    3      25.291103  0.063320  0.048507  
    4      25.096743  0.063745  0.063539  
    ...          ...       ...       ...  
    99995  24.737946  0.165555  0.106471  
    99996  24.224169  0.068703  0.053617  
    99997  25.613836  0.031766  0.028188  
    99998  25.274899  0.014528  0.010207  
    99999  25.699642  0.064083  0.048538  
    
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
          <td>27.284061</td>
          <td>0.670312</td>
          <td>26.585146</td>
          <td>0.149012</td>
          <td>26.023188</td>
          <td>0.080477</td>
          <td>25.188973</td>
          <td>0.062780</td>
          <td>24.657718</td>
          <td>0.075060</td>
          <td>23.990544</td>
          <td>0.093731</td>
          <td>0.161398</td>
          <td>0.120450</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.331255</td>
          <td>1.276210</td>
          <td>27.738821</td>
          <td>0.384492</td>
          <td>26.627852</td>
          <td>0.136510</td>
          <td>26.088045</td>
          <td>0.138142</td>
          <td>25.765087</td>
          <td>0.195943</td>
          <td>25.611141</td>
          <td>0.365274</td>
          <td>0.082764</td>
          <td>0.079149</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.614610</td>
          <td>0.725971</td>
          <td>27.685697</td>
          <td>0.329414</td>
          <td>26.147135</td>
          <td>0.145355</td>
          <td>25.204295</td>
          <td>0.121234</td>
          <td>24.411566</td>
          <td>0.135276</td>
          <td>0.150925</td>
          <td>0.120204</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.470702</td>
          <td>1.374463</td>
          <td>29.315185</td>
          <td>1.123266</td>
          <td>27.950347</td>
          <td>0.405090</td>
          <td>25.998885</td>
          <td>0.127893</td>
          <td>25.505420</td>
          <td>0.157186</td>
          <td>24.991308</td>
          <td>0.221310</td>
          <td>0.063320</td>
          <td>0.048507</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.374150</td>
          <td>0.341655</td>
          <td>25.989198</td>
          <td>0.088756</td>
          <td>26.188500</td>
          <td>0.093086</td>
          <td>25.639621</td>
          <td>0.093470</td>
          <td>25.516074</td>
          <td>0.158625</td>
          <td>25.025152</td>
          <td>0.227622</td>
          <td>0.063745</td>
          <td>0.063539</td>
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
          <td>26.613562</td>
          <td>0.411536</td>
          <td>26.385951</td>
          <td>0.125494</td>
          <td>25.448348</td>
          <td>0.048353</td>
          <td>25.063337</td>
          <td>0.056157</td>
          <td>24.905164</td>
          <td>0.093349</td>
          <td>24.399295</td>
          <td>0.133850</td>
          <td>0.165555</td>
          <td>0.106471</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.342053</td>
          <td>0.333099</td>
          <td>26.705936</td>
          <td>0.165226</td>
          <td>25.929701</td>
          <td>0.074098</td>
          <td>25.176878</td>
          <td>0.062110</td>
          <td>24.731803</td>
          <td>0.080135</td>
          <td>24.323121</td>
          <td>0.125308</td>
          <td>0.068703</td>
          <td>0.053617</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.248228</td>
          <td>0.309135</td>
          <td>26.902919</td>
          <td>0.195218</td>
          <td>26.155557</td>
          <td>0.090430</td>
          <td>25.926193</td>
          <td>0.120075</td>
          <td>25.810028</td>
          <td>0.203482</td>
          <td>25.718139</td>
          <td>0.396913</td>
          <td>0.031766</td>
          <td>0.028188</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.675476</td>
          <td>0.193115</td>
          <td>26.201123</td>
          <td>0.106859</td>
          <td>26.172372</td>
          <td>0.091776</td>
          <td>25.982462</td>
          <td>0.126086</td>
          <td>26.018785</td>
          <td>0.242069</td>
          <td>25.385828</td>
          <td>0.305571</td>
          <td>0.014528</td>
          <td>0.010207</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.543649</td>
          <td>0.389994</td>
          <td>26.629587</td>
          <td>0.154796</td>
          <td>26.646051</td>
          <td>0.138671</td>
          <td>26.253003</td>
          <td>0.159169</td>
          <td>26.447901</td>
          <td>0.342356</td>
          <td>25.620470</td>
          <td>0.367946</td>
          <td>0.064083</td>
          <td>0.048538</td>
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
          <td>26.645509</td>
          <td>0.486440</td>
          <td>26.784224</td>
          <td>0.214127</td>
          <td>26.133636</td>
          <td>0.111124</td>
          <td>25.265521</td>
          <td>0.085107</td>
          <td>24.886494</td>
          <td>0.114981</td>
          <td>23.942293</td>
          <td>0.113326</td>
          <td>0.161398</td>
          <td>0.120450</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.469739</td>
          <td>1.485199</td>
          <td>27.954764</td>
          <td>0.520143</td>
          <td>26.263047</td>
          <td>0.119253</td>
          <td>26.632738</td>
          <td>0.262423</td>
          <td>26.059788</td>
          <td>0.296642</td>
          <td>25.096339</td>
          <td>0.287877</td>
          <td>0.082764</td>
          <td>0.079149</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.386151</td>
          <td>1.241603</td>
          <td>25.991722</td>
          <td>0.159205</td>
          <td>24.960944</td>
          <td>0.122083</td>
          <td>24.309132</td>
          <td>0.154835</td>
          <td>0.150925</td>
          <td>0.120204</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.602811</td>
          <td>0.807774</td>
          <td>27.330334</td>
          <td>0.289926</td>
          <td>26.429858</td>
          <td>0.219586</td>
          <td>25.749311</td>
          <td>0.227752</td>
          <td>24.918412</td>
          <td>0.246357</td>
          <td>0.063320</td>
          <td>0.048507</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.137401</td>
          <td>0.669652</td>
          <td>26.135887</td>
          <td>0.117773</td>
          <td>25.736046</td>
          <td>0.074493</td>
          <td>25.682176</td>
          <td>0.116351</td>
          <td>25.632187</td>
          <td>0.207138</td>
          <td>25.346655</td>
          <td>0.348830</td>
          <td>0.063745</td>
          <td>0.063539</td>
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
          <td>26.204745</td>
          <td>0.130477</td>
          <td>25.347491</td>
          <td>0.055409</td>
          <td>24.985072</td>
          <td>0.066227</td>
          <td>25.053191</td>
          <td>0.132477</td>
          <td>24.626038</td>
          <td>0.202923</td>
          <td>0.165555</td>
          <td>0.106471</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.579576</td>
          <td>0.895054</td>
          <td>26.755935</td>
          <td>0.200002</td>
          <td>26.171647</td>
          <td>0.109139</td>
          <td>25.287606</td>
          <td>0.082267</td>
          <td>24.806580</td>
          <td>0.101885</td>
          <td>24.494694</td>
          <td>0.173151</td>
          <td>0.068703</td>
          <td>0.053617</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.870848</td>
          <td>0.218341</td>
          <td>26.239615</td>
          <td>0.114692</td>
          <td>26.287061</td>
          <td>0.193368</td>
          <td>25.949088</td>
          <td>0.266532</td>
          <td>25.802402</td>
          <td>0.489663</td>
          <td>0.031766</td>
          <td>0.028188</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.249166</td>
          <td>0.344101</td>
          <td>26.340723</td>
          <td>0.138979</td>
          <td>25.898715</td>
          <td>0.084865</td>
          <td>25.824449</td>
          <td>0.129893</td>
          <td>25.810912</td>
          <td>0.237384</td>
          <td>24.917258</td>
          <td>0.243722</td>
          <td>0.014528</td>
          <td>0.010207</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.128765</td>
          <td>0.664604</td>
          <td>26.585671</td>
          <td>0.172941</td>
          <td>26.689783</td>
          <td>0.170324</td>
          <td>26.343792</td>
          <td>0.204384</td>
          <td>26.142623</td>
          <td>0.313869</td>
          <td>25.814075</td>
          <td>0.497259</td>
          <td>0.064083</td>
          <td>0.048538</td>
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
          <td>29.143936</td>
          <td>2.047208</td>
          <td>26.446130</td>
          <td>0.159468</td>
          <td>25.912380</td>
          <td>0.090578</td>
          <td>25.344545</td>
          <td>0.090219</td>
          <td>24.871716</td>
          <td>0.112308</td>
          <td>23.915114</td>
          <td>0.109463</td>
          <td>0.161398</td>
          <td>0.120450</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.706769</td>
          <td>1.599406</td>
          <td>27.047366</td>
          <td>0.235703</td>
          <td>26.590863</td>
          <td>0.143267</td>
          <td>26.415139</td>
          <td>0.198335</td>
          <td>25.752543</td>
          <td>0.209606</td>
          <td>25.109666</td>
          <td>0.264175</td>
          <td>0.082764</td>
          <td>0.079149</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.374199</td>
          <td>1.426695</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.696538</td>
          <td>0.397524</td>
          <td>26.058083</td>
          <td>0.165245</td>
          <td>24.843160</td>
          <td>0.108096</td>
          <td>24.315364</td>
          <td>0.152682</td>
          <td>0.150925</td>
          <td>0.120204</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.025427</td>
          <td>1.825864</td>
          <td>28.031625</td>
          <td>0.495088</td>
          <td>27.327509</td>
          <td>0.256243</td>
          <td>26.127815</td>
          <td>0.149143</td>
          <td>25.433223</td>
          <td>0.153824</td>
          <td>25.529340</td>
          <td>0.355936</td>
          <td>0.063320</td>
          <td>0.048507</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.986977</td>
          <td>0.560162</td>
          <td>26.046679</td>
          <td>0.097693</td>
          <td>25.976205</td>
          <td>0.081363</td>
          <td>25.727242</td>
          <td>0.106581</td>
          <td>25.363964</td>
          <td>0.146487</td>
          <td>26.710280</td>
          <td>0.838305</td>
          <td>0.063745</td>
          <td>0.063539</td>
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
          <td>26.465832</td>
          <td>0.160705</td>
          <td>25.417424</td>
          <td>0.057874</td>
          <td>25.002477</td>
          <td>0.065977</td>
          <td>24.704255</td>
          <td>0.096001</td>
          <td>24.881864</td>
          <td>0.246557</td>
          <td>0.165555</td>
          <td>0.106471</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.999115</td>
          <td>1.082395</td>
          <td>26.931982</td>
          <td>0.208354</td>
          <td>26.003837</td>
          <td>0.083099</td>
          <td>25.119174</td>
          <td>0.062149</td>
          <td>24.842683</td>
          <td>0.092790</td>
          <td>24.132523</td>
          <td>0.111637</td>
          <td>0.068703</td>
          <td>0.053617</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.307595</td>
          <td>0.326592</td>
          <td>26.818967</td>
          <td>0.183719</td>
          <td>26.352223</td>
          <td>0.108745</td>
          <td>26.343986</td>
          <td>0.174135</td>
          <td>25.733118</td>
          <td>0.192961</td>
          <td>25.701475</td>
          <td>0.396191</td>
          <td>0.031766</td>
          <td>0.028188</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.106927</td>
          <td>0.276251</td>
          <td>26.122597</td>
          <td>0.099956</td>
          <td>26.121842</td>
          <td>0.087975</td>
          <td>25.760216</td>
          <td>0.104126</td>
          <td>25.707412</td>
          <td>0.187026</td>
          <td>24.913238</td>
          <td>0.207787</td>
          <td>0.014528</td>
          <td>0.010207</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.925339</td>
          <td>0.532562</td>
          <td>26.764213</td>
          <td>0.179856</td>
          <td>26.708534</td>
          <td>0.152452</td>
          <td>26.083057</td>
          <td>0.143604</td>
          <td>25.842113</td>
          <td>0.217505</td>
          <td>26.122798</td>
          <td>0.557048</td>
          <td>0.064083</td>
          <td>0.048538</td>
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
