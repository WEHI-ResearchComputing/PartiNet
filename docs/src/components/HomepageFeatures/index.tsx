import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Easy to Use',
    Svg: require('@site/static/img/undraw_data-analysis_b7cp.svg').default,
    description: (
      <>
        PartiNet is a robust, automated picking pipeline without the need for iterative retraining or user-supplied templates or box sizes.
      </>
    ),
  },
  {
    title: 'High quality picking',
    Svg: require('@site/static/img/undraw_success_288d.svg').default,
    description: (
      <>
        PartiNet proides improved identification of rare particle views allowing for comprehensive map reconstruction.
      </>
    ),
  },
  {
    title: 'Fast picking',
    Svg: require('@site/static/img/undraw_space-exploration_dhu1.svg').default,
    description: (
      <>
        PartiNet provides up to 10× faster inference than existing tools enabling real-time, on-the-fly picking. 
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
