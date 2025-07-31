import { NavigationBarSection } from "./screens/RealPage/sections/NavigationBarSection/index";
import { RecommendedProductsSection } from "./screens/RealPage/sections/RecommendedProductsSection/index";
import { FooterSection } from './screens/RealPage/sections/FooterSection/index';


function App() {
    return (
        <div className="min-h-screen bg-white">
            <NavigationBarSection />
            <RecommendedProductsSection />
            <FooterSection />
        </div>
    );
}

export default App;
